/**
 * @file test_lme_annulus.cpp
 * @brief LME FEM vs Leissa Table 2.18 — annular plate, w = 0 pinned
 *        on both rings (partial clamp).
 *
 * Port of @ref tests/shell/test_modes_vs_annulus_clamped_analytic.cpp
 * onto @ref chladni::shell::LMEAssembler.
 *
 * @section setup Setup
 *  - Annulus: outer @f$ a = 0.10 @f$ m, inner @f$ b = 0.05 @f$ m
 *    (@f$ b/a = 0.5 @f$ — Leissa's tabulation column).
 *  - Steel: @f$ E = 200 @f$ GPa, @f$ \nu = 1/3 @f$,
 *    @f$ \rho = 7850 @f$ kg/m³.
 *  - Thickness @f$ h = 1 @f$ mm.
 *
 * @section bc Boundary condition — w = 0 only (partial clamp)
 *
 * This test pins the z-DOF of every boundary vertex (both rings) and
 * leaves the in-plane and one-ring-inward DOFs free. With the LME
 * basis's weak Kronecker-delta property — interior-node basis
 * functions vanish on boundary edges (Arroyo--Ortiz §3.1) — the
 * z-DOF pin enforces @f$ w \equiv 0 @f$ on the rim **exactly**.
 *
 * It does **not** enforce the normal-derivative condition
 * @f$ \partial w / \partial n = 0 @f$ that distinguishes the
 * fully-clamped Kirchhoff plate from the simply-supported one. True
 * clamped BC for an LME basis would need either a ghost layer of nodes outside
 * @f$ \partial\Omega @f$ or a one-ring of inward-projected fictitious
 * DOFs. Neither is in scope yet.
 *
 * The Loop pipeline's equivalent of this trick (zero the z-component
 * of the third vertex of every boundary face) is part of
 * @c test_modes_vs_annulus_clamped_analytic.cpp; an analogous LME
 * construction is a planned follow-up.
 *
 * @section ref What we expect to see
 *
 * We compare against Leissa Table 2.18's **clamped-clamped** values
 * (the only annular analytic the project ships). With only the @c w=0
 * half of the clamped pair enforced, the FEM frequencies land
 * **below** the CC reference — the rim is free to rotate, behaving as
 * simply-supported. The mismatch is therefore physical, not numerical,
 * and is the dominant signal in @c rel_err.
 *
 * Observed at 32 × 8 (256 V):
 *
 *   Mode        LME (Hz)    Leissa CC (Hz)    signed rel_err
 *   (n=0, s=0)  1043.8      2195.3            −52.5 %
 *   (n=1, s=0)  1087.3      2219.4            −51.0 %  (doublet mean)
 *   (n=2, s=0)  1222.7      2295.4            −46.7 %  (doublet mean)
 *
 * All three modes land at ~half the CC frequency, consistent with the
 * SS-SS-vs-CC-CC stiffness ratio for thin annular plates at b/a=0.5
 * (the CC additional rotation constraint roughly doubles the
 * eigenvalue at this aspect ratio). When the ghost-layer LME clamp
 * lands (§6 follow-up) those numbers should rise toward the CC line.
 *
 * Tolerances below are sized to lock in current behaviour at the
 * **partial-clamp ~50 % gap** level. The cluster-split structural
 * check still applies and stays sub-2 % at this resolution
 * (rotationally symmetric polar mesh).
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include "lme_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

using chladni::shell::LMEAssembler;
using chladni::shell::ShellMaterial;
using chladni::tests::lme::solve_lme_z_modes;

namespace {

chladni::IsotropicMaterial steel_third()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 1.0 / 3.0,
            .density        = 7850.0};
}

}  // namespace

TEST_CASE("LME FEM vs Leissa: annular plate, w=0 pinned (partial clamp)",
          "[lme][assembler][modes][plate][annulus][analytic][validation]")
{
    constexpr double a = 0.10;   // outer radius
    constexpr double b = 0.05;   // inner radius
    constexpr double h = 1.0e-3;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 8;
    constexpr std::size_t n_modes = 5;

    const auto mat  = steel_third();
    const auto mesh = chladni::mesh::generate_annulus(a, b, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();

    const auto analytic =
        chladni::analytical::annular_plate_clamped_clamped_angular_frequencies(
            {.radius_outer = a, .radius_inner = b, .thickness = h},
            mat, n_modes);
    REQUIRE(analytic.size() == n_modes);

    // Boundary detection — both rings via the edge-list boundary flag.
    const auto edges = chladni::shell::build_edges(mesh.F);
    std::vector<bool> is_bdry(static_cast<std::size_t>(n_v), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }

    LMEAssembler::Params p;
    // Flat path — Dirichlet-pinning helper assumes 3·n_v K shape.
    p.use_curved_shell = false;
    p.newton_tol       = 1e-13;
    p.newton_max_iters = 60;
    const auto sm = chladni::shell::shell_material_from_isotropic(mat, h);

    const auto modes = solve_lme_z_modes(
        mesh.V, mesh.F, sm, mat.density * h, is_bdry, n_modes, p);

    // Spectrum: no rigid modes survive (both rings z-pinned → K_red
    // positive-definite). Compare directly.
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    std::vector<double> fem_hz(n_modes);
    for (std::size_t k = 0; k < n_modes; ++k) {
        REQUIRE(modes.eigenvalues(static_cast<Eigen::Index>(k)) > 0.0);
        fem_hz[k] = std::sqrt(modes.eigenvalues(
            static_cast<Eigen::Index>(k))) / two_pi;
    }

    std::cout << "[LME annulus a=" << a << " b=" << b << " h=" << h
              << "  V=" << n_v << "  (w=0 pin, no ∂w/∂n pin)]\n"
              << "  Leissa CC (Hz):";
    for (double w : analytic) std::cout << " " << w / two_pi;
    std::cout << "\n  LME       (Hz):";
    for (double f : fem_hz) std::cout << " " << f;
    std::cout << '\n';

    auto rel_signed = [](double f, double f_ref) {
        return (f - f_ref) / f_ref;
    };

    // Partial clamp (w=0 only) consistently lands at ~half the CC
    // frequency. The 55 % envelope locks in current SS-SS-style
    // behaviour; bring it down once ∂w/∂n = 0 enforcement lands.
    INFO("(n=0, s=0) LME = " << fem_hz[0]
         << " Hz, Leissa CC = " << analytic[0] / two_pi
         << " Hz, signed_rel_err = " << rel_signed(fem_hz[0], analytic[0] / two_pi));
    REQUIRE(std::abs(rel_signed(fem_hz[0], analytic[0] / two_pi)) < 0.55);

    const double f_n1 = 0.5 * (fem_hz[1] + fem_hz[2]);
    INFO("(n=1, s=0) LME mean = " << f_n1
         << " Hz, Leissa CC = " << analytic[1] / two_pi
         << " Hz, signed_rel_err = " << rel_signed(f_n1, analytic[1] / two_pi));
    REQUIRE(std::abs(rel_signed(f_n1, analytic[1] / two_pi)) < 0.55);

    const double f_n2 = 0.5 * (fem_hz[3] + fem_hz[4]);
    INFO("(n=2, s=0) LME mean = " << f_n2
         << " Hz, Leissa CC = " << analytic[3] / two_pi
         << " Hz, signed_rel_err = " << rel_signed(f_n2, analytic[3] / two_pi));
    REQUIRE(std::abs(rel_signed(f_n2, analytic[3] / two_pi)) < 0.55);

    // Cluster-split structural sanity. Polar annulus is rotationally
    // symmetric so n=1 cos/sin should be very close; a non-trivial
    // split would indicate a node-spacing or quadrature symmetry bug.
    REQUIRE(std::abs(fem_hz[2] - fem_hz[1]) / f_n1 < 0.02);
    REQUIRE(std::abs(fem_hz[4] - fem_hz[3]) / f_n2 < 0.02);

    // Monotonicity sanity: each mode is at or above the previous.
    for (std::size_t k = 0; k + 1 < n_modes; ++k) {
        REQUIRE(fem_hz[k + 1] >= fem_hz[k] - 1e-9);
    }
}
