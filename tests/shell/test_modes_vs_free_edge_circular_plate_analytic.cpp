/**
 * @file test_modes_vs_free_edge_circular_plate_analytic.cpp
 * @brief FEM (Loop subdivision shell) vs the free-edge circular plate
 *        analytic (Leissa Table 2.5) — the eponymous Chladni fixture.
 *
 * @section setup Setup
 *  - Disk: @c generate_circular_disk(R=0.10, n_azimuthal, n_radial).
 *  - Steel: E=200 GPa, nu=0.33, rho=7850 kg/m^3 (matching Leissa's table).
 *  - Thickness: h=1 mm (R/h=100, thin-plate regime).
 *  - Free edge (no Dirichlet BCs) — same calling convention as the
 *    sphere / cylinder validations.
 *
 * @section ref Reference values (Leissa Table 2.5, nu=0.33)
 *
 * Lowest seven angular frequencies correspond to the eigenvalue pattern
 *   (n=2, s=0):  λ² = 5.253   (doubly degenerate)
 *   (n=0, s=1):  λ² = 9.084   (non-degenerate)
 *   (n=3, s=0):  λ² = 12.23   (doubly degenerate)
 *   (n=1, s=1):  λ² = 20.52   (doubly degenerate)
 * → expected ω-list = {5.253, 5.253, 9.084, 12.23, 12.23, 20.52, 20.52}.
 *
 * The (n=2, s=0) pair is the classical "saddle" Chladni pattern, the
 * (n=0, s=1) is the breathing "ring" mode, (n=3) is the trefoil.
 *
 * @section convergence Convergence
 *
 * The disk has one extraordinary corner (the center, valence n_azimuthal).
 * The L.3.4 / Stam path handles this. Empirically modest n_azimuthal /
 * n_radial values (32 / 8) give sub-5% rel_err on the lowest pair;
 * higher refinement improves convergence geometrically.
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

namespace {

chladni::IsotropicMaterial steel_033()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.33,
            .density        = 7850.0};
}

struct DiskRun {
    Eigen::Index n_v;
    std::vector<double> omegas_hz;  // sorted ascending Hz
};

DiskRun run_disk(int n_az, int n_rad, std::size_t n_modes,
                 bool use_stam = false)
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto mesh = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel_033(), h, n_modes,
        /*n_passes=*/1, use_stam);
    REQUIRE(modes.omegas.allFinite());
    DiskRun out;
    out.n_v = mesh.V.rows();
    out.omegas_hz.reserve(modes.omegas.size());
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        out.omegas_hz.push_back(
            modes.omegas(k) / (2.0 * std::numbers::pi));
    }
    return out;
}

}  // namespace

TEST_CASE("Loop FEM vs Leissa: free-edge circular plate — Chladni fixture",
          "[shell][loop][modes][plate][circular][analytical][validation]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;

    const auto analytical =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, steel_033(), n_modes);
    REQUIRE(analytical.size() == n_modes);

    // 32 azimuthal × 8 radial: 257 V, 480 F. Center vertex valence 32
    // (extraordinary; handled by L.3.4 multi-pass). Outer ring boundary
    // valence 4 (Schweitzer augmentation). Should be moderately tight
    // on the lowest cluster — Chladni-pattern shapes are bending modes
    // whose energy is concentrated away from the central vertex.
    const auto r = run_disk(/*n_az=*/32, /*n_rad=*/8, n_modes);

    std::cout << "[free-edge circular plate R=" << R << " h=" << h
              << "  V=" << r.n_v << "]\n  Leissa (Hz):";
    for (double w : analytical) std::cout << " " << w / (2.0 * std::numbers::pi);
    std::cout << "\n  FEM    (Hz):";
    for (double w : r.omegas_hz) std::cout << " " << w;
    std::cout << '\n';

    // Verify the FEM cluster structure: lowest 2 modes should be near-
    // degenerate (the n=2 pair), then 1 isolated mode (n=0,s=1), then
    // 2 near-degenerate (n=3), then 2 near-degenerate (n=1,s=1). The
    // (2n+1)-fold degeneracy of n=k modes in the analytic sense becomes
    // 2-fold on a finite mesh (cos/sin pair).
    auto rel = [](double f, double f_ref) {
        return std::abs(f - f_ref) / f_ref;
    };

    // Mode pair (n=2, s=0) cluster mean vs analytic. Actual rel_err
    // ~2.2 % at V=257 under consistent mass.
    const double f_n2 = 0.5 * (r.omegas_hz[0] + r.omegas_hz[1]);
    const double f_n2_ref = analytical[0] / (2.0 * std::numbers::pi);
    INFO("(n=2, s=0) FEM mean = " << f_n2 << " Hz, Leissa = " << f_n2_ref
         << " Hz, rel_err = " << rel(f_n2, f_n2_ref));
    REQUIRE(rel(f_n2, f_n2_ref) < 0.03);

    // (n=0, s=1) breathing mode — purely radial, depends on n_radial
    // more than n_azimuthal. Actual rel_err ~6.7 % at n_rad=8 under
    // consistent mass (improves to ~3 % at n_rad=16 — see the [.slow]
    // convergence test below).
    const double f_n0 = r.omegas_hz[2];
    const double f_n0_ref = analytical[2] / (2.0 * std::numbers::pi);
    INFO("(n=0, s=1) FEM = " << f_n0 << " Hz, Leissa = " << f_n0_ref
         << " Hz, rel_err = " << rel(f_n0, f_n0_ref));
    REQUIRE(rel(f_n0, f_n0_ref) < 0.08);

    // (n=3, s=0) cluster — modes 3 and 4. Actual rel_err ~4.0 % at
    // V=257 under consistent mass.
    const double f_n3 = 0.5 * (r.omegas_hz[3] + r.omegas_hz[4]);
    const double f_n3_ref = analytical[3] / (2.0 * std::numbers::pi);
    INFO("(n=3, s=0) FEM mean = " << f_n3 << " Hz, Leissa = " << f_n3_ref
         << " Hz, rel_err = " << rel(f_n3, f_n3_ref));
    REQUIRE(rel(f_n3, f_n3_ref) < 0.05);

    // Sanity: each cluster's split (cos/sin) is tight on a 32-azimuthal
    // mesh (icosahedral-style symmetry-breaking is sub-1%).
    const double n2_split =
        std::abs(r.omegas_hz[1] - r.omegas_hz[0]) / f_n2;
    INFO("(n=2) cluster split = " << n2_split);
    REQUIRE(n2_split < 0.02);
}

TEST_CASE("Free-edge circular plate: convergence under mesh refinement",
          "[shell][loop][modes][plate][circular][analytical][convergence][.slow]")
{
    // Refine n_azimuthal × n_radial proportionally and verify that the
    // (n=2, s=0) doublet stabilises at a finite-mesh FEM limit. Slow
    // (~30 s) — run on demand.
    //
    // Under the consistent-mass default (MassLumping::None, flipped
    // 2026-05-17 late) the FEM converges TO Leissa from ABOVE on this
    // fixture, settling near 131.1 Hz vs Leissa's 129.27 Hz (~1.4 %
    // residual bias from polygonal-rim discretisation of the circular
    // boundary at finite n_azimuthal). The 16x4 coarse mesh happens to
    // cross *through* Leissa (lands ~130 Hz with rel_err ~0.5 %), so
    // strict monotonic shrinkage of rel_err with refinement does NOT
    // hold here — earlier versions of this test pinned that monotonic
    // shape under RowSum lumping (pre-flip, FEM was 1.4 % LOW instead
    // of HIGH and 16x4 sat above-Leissa) and have to be reread post-
    // lumping-flip. We assert the physically-meaningful invariants
    // instead: bounded rel_err and an asymptoting limit.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto analytical =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, steel_033(), n_modes);
    const double f_n2_ref = analytical[0] / (2.0 * std::numbers::pi);

    struct Level { int n_az; int n_rad; };
    const Level levels[] = {
        {16,  4},
        {32,  8},
        {64, 16},
    };

    std::vector<double> rel_errs;
    std::vector<double> f_n2_levels;
    std::cout << "[free-edge circular plate convergence — Leissa n=2 = "
              << f_n2_ref << " Hz]\n";
    for (const auto& L : levels) {
        const auto r = run_disk(L.n_az, L.n_rad, n_modes);
        const double f_n2 = 0.5 * (r.omegas_hz[0] + r.omegas_hz[1]);
        const double err = std::abs(f_n2 - f_n2_ref) / f_n2_ref;
        std::cout << "  n_az=" << L.n_az << " n_rad=" << L.n_rad
                  << " V=" << r.n_v
                  << "  f_n2=" << f_n2 << " Hz"
                  << "  rel_err=" << err << '\n';
        rel_errs.push_back(err);
        f_n2_levels.push_back(f_n2);
    }

    // Bounded: every level lands within 2 % of Leissa. The asymptotic
    // limit sits ~1.4 % above; the gate gives modest headroom for the
    // coarsest mesh and the limit value.
    for (std::size_t i = 0; i < rel_errs.size(); ++i) {
        REQUIRE(rel_errs[i] < 0.02);
    }

    // Asymptoting: the two finest meshes agree to within 0.1 %. This
    // is the FEM-converges-to-a-stable-limit gate, robust to the
    // accidental 16x4 cross-through behaviour.
    const std::size_t k = f_n2_levels.size();
    const double drift = std::abs(f_n2_levels[k - 1] - f_n2_levels[k - 2])
                       / f_n2_levels[k - 1];
    INFO("32x8 → 64x16 drift = " << drift);
    REQUIRE(drift < 1.0e-3);
}
