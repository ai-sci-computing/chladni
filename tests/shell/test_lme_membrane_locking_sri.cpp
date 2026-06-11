/**
 * @file test_lme_membrane_locking_sri.cpp
 * @brief R10 scoped experiment: does reduced-order integration relieve the
 *        free-free cylinder membrane locking on the meshfree LME path?
 *
 * The cylinder n=2 ovalling mode is bending-dominated, but the meshfree LME
 * curved-shell membrane term over-stiffens it (genuine membrane locking, HF
 * membrane fraction 0.444 vs Loop 0.039 at aspect 2.39 — see
 * [lme_cylinder_locking]). Selective/reduced integration is the classic FEM
 * cure. Before wiring a membrane-only reduced rule into the core assembler,
 * this probe runs the cheap precursor: sweep the UNIFORM Galerkin quadrature
 * order (n_quadrature_per_tri ∈ {1,3,7,12}) and measure the n=2 error vs the
 * Rayleigh-Love analytic. If a lower order relieves the lock, membrane-only
 * SRI is worth implementing; if not, locking is integration-insensitive and
 * the experiment documents that.
 *
 * RESULT (2026-06-09), R=0.10 L=0.30 h=1mm cylinder, 20x6:
 *   Loop  +6.0 %   (does not lock — validates mesh/analytic/extraction)
 *   SME   +23.2 %  (the shipped curved default's documented lock)
 *   1st-order LME n=2 error vs uniform quadrature order:
 *     n_quad=1  +5671 %   n_quad=3  +5719 %
 *     n_quad=7  +5317 %   n_quad=12 +5297 %
 * Going from the full 12-point rule to 1-point moves the error by ~7 %
 * relative against a +5300 % lock — the lock is INTEGRATION-INSENSITIVE.
 * Membrane-only selective reduced integration (which would perturb the
 * result even less, keeping bending at full order) therefore cannot relieve
 * it: the meshfree LME/SME membrane strain is non-zero for the inextensional
 * ovalling mode regardless of quadrature, so no integration scheme recovers
 * it. Decision: do NOT implement SRI; the locking stays a documented
 * Weakness (it is why the Loop path ships for membrane-dominated curved
 * shells — see [lme_cylinder_locking]).
 *
 * Run: SRI_DIAG=1 ./chladni_tests "[sri_probe]".
 */

#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

chladni::IsotropicMaterial steel()
{
    return chladni::IsotropicMaterial{
        .youngs_modulus = 200.0e9, .poisson_ratio = 0.30, .density = 7850.0};
}

/// Dominant circumferential wavenumber of a mode shape (radial-displacement
/// Fourier projection over rings), assuming generate_cylinder's
/// j*n_around+i vertex ordering.
int dominant_n(const Eigen::VectorXd& u, int n_around, int n_along)
{
    constexpr int n_max = 6;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    std::array<double, n_max + 1> Es{};
    for (int n = 0; n <= n_max; ++n) {
        double total = 0.0;
        for (int j = 0; j <= n_along; ++j) {
            double c = 0.0, s = 0.0;
            for (int i = 0; i < n_around; ++i) {
                const Eigen::Index v = j * n_around + i;
                const double phi = two_pi * i / n_around;
                const double u_r = u(3 * v + 0) * std::cos(phi)
                                 + u(3 * v + 1) * std::sin(phi);
                c += u_r * std::cos(n * phi);
                s += u_r * std::sin(n * phi);
            }
            total += c * c + s * s;
        }
        Es[static_cast<std::size_t>(n)] = total;
    }
    int best = 0;
    for (int n = 1; n <= n_max; ++n)
        if (Es[static_cast<std::size_t>(n)] > Es[static_cast<std::size_t>(best)])
            best = n;
    return best;
}

}  // namespace

TEST_CASE("R10 probe: LME cylinder n=2 locking vs uniform quadrature order",
          "[.diag][shell][lme][cylinder][sri_probe]")
{
    const bool diag = std::getenv("SRI_DIAG") != nullptr;

    constexpr double R = 0.10, L = 0.30, h = 1.0e-3;
    constexpr int n_around = 20, n_along = 6;
    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const auto analytic = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, steel(), /*n_circumferential=*/3);  // n = 2, 3, 4
    const double w2_analytic = analytic[0];

    const auto sm = chladni::shell::shell_material_from_isotropic(steel(), h);

    // Pipeline validation: Loop on the same mesh should land within a few
    // percent of analytic (it does not lock), confirming the mesh ordering,
    // analytic reference and dominant_n extraction are sound.
    {
        const auto loop_modes = chladni::shell::compute_shell_modes_loop(
            mesh.V, mesh.F, steel(), h, /*n_modes=*/14);
        double w2_loop = -1.0;
        for (Eigen::Index k = 0; k < loop_modes.omegas.size(); ++k) {
            if (dominant_n(loop_modes.shapes.col(k), n_around, n_along) == 2) {
                w2_loop = loop_modes.omegas(k);
                break;
            }
        }
        if (diag) {
            std::printf("[sri_probe] LOOP baseline  w2_fem=%.2f  w2_an=%.2f  "
                        "rel_err=%+.1f%%\n",
                        w2_loop, w2_analytic,
                        100.0 * (w2_loop - w2_analytic) / w2_analytic);
        }
    }

    // SME (the shipped curved default) baseline at the full rule.
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        const chladni::shell::LMEAssembler sme{p};
        double w2_sme = -1.0;
        try {
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, steel(), sm, h, 14, sme);
            for (Eigen::Index k = 0; k < m.omegas.size(); ++k)
                if (dominant_n(m.shapes.col(k), n_around, n_along) == 2) {
                    w2_sme = m.omegas(k); break;
                }
        } catch (const std::exception& e) {
            if (diag) std::printf("[sri_probe] SME threw: %s\n", e.what());
        }
        if (diag) {
            std::printf("[sri_probe] SME baseline   w2_fem=%.2f  w2_an=%.2f  "
                        "rel_err=%+.1f%%\n",
                        w2_sme, w2_analytic,
                        100.0 * (w2_sme - w2_analytic) / w2_analytic);
        }
    }

    // GAMMA SWEEP (user observation 2026-06-09): on the cylinder the basis
    // support width γ — not the quadrature order — may be the locking lever.
    // Sharper γ (smaller support) behaves more like Loop's compact stencil.
    if (diag) {
        std::printf("[sri_probe] --- 1st-order LME n=2 vs gamma (fixed 12-pt) ---\n");
        for (double g : {0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0}) {
            chladni::shell::LMEAssembler::Params p;
            p.gamma = g;
            const chladni::shell::LMEAssembler asm_g{p};
            double w2 = -1.0;
            try {
                const auto m = chladni::shell::compute_shell_modes(
                    mesh.V, mesh.F, steel(), sm, h, 14, asm_g);
                for (Eigen::Index k = 0; k < m.omegas.size(); ++k)
                    if (dominant_n(m.shapes.col(k), n_around, n_along) == 2) {
                        w2 = m.omegas(k); break;
                    }
            } catch (const std::exception& e) {
                std::printf("[sri_probe] gamma=%.1f threw: %s\n", g, e.what());
            }
            std::printf("[sri_probe] gamma=%.1f  w2_fem=%.2f  rel_err=%+.1f%%\n",
                        g, w2, 100.0 * (w2 - w2_analytic) / w2_analytic);
        }
        std::printf("[sri_probe] --- (SME gamma sweep) ---\n");
        for (double g : {0.8, 2.5, 4.0}) {
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            p.gamma = g;
            const chladni::shell::LMEAssembler asm_g{p};
            double w2 = -1.0;
            try {
                const auto m = chladni::shell::compute_shell_modes(
                    mesh.V, mesh.F, steel(), sm, h, 14, asm_g);
                for (Eigen::Index k = 0; k < m.omegas.size(); ++k)
                    if (dominant_n(m.shapes.col(k), n_around, n_along) == 2) {
                        w2 = m.omegas(k); break;
                    }
            } catch (const std::exception& e) {
                std::printf("[sri_probe] SME gamma=%.1f threw: %s\n", g, e.what());
            }
            std::printf("[sri_probe] SME gamma=%.1f  w2_fem=%.2f  rel_err=%+.1f%%\n",
                        g, w2, 100.0 * (w2 - w2_analytic) / w2_analytic);
        }
    }

    std::vector<double> errs;
    for (int nq : {1, 3, 7, 12}) {
        chladni::shell::LMEAssembler::Params p;  // 1st-order LME, curved, ghost
        p.n_quadrature_per_tri = nq;
        const chladni::shell::LMEAssembler assembler{p};

        double w2 = -1.0;
        try {
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, steel(), sm, h, /*n_modes=*/14, assembler);
            for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
                if (dominant_n(modes.shapes.col(k), n_around, n_along) == 2) {
                    w2 = modes.omegas(k);
                    break;
                }
            }
        } catch (const std::exception& e) {
            if (diag) std::printf("[sri_probe] nq=%d threw: %s\n", nq, e.what());
        }
        const double err = (w2 > 0.0)
            ? (w2 - w2_analytic) / w2_analytic : std::nan("");
        errs.push_back(err);
        if (diag) {
            std::printf("[sri_probe] n_quad=%2d  w2_fem=%.2f  w2_an=%.2f  "
                        "rel_err=%+.1f%%\n",
                        nq, w2, w2_analytic, 100.0 * err);
        }
    }

    // Guard the conclusion: every quadrature order locks catastrophically
    // (n=2 error > +100 %) AND the spread across orders is small relative to
    // the lock (< 30 %), i.e. the lock is integration-insensitive so reduced
    // integration cannot cure it. If this ever breaks, re-open the SRI
    // question — the locking would have become integration-sensitive.
    REQUIRE(errs.size() == 4);
    double lo = errs[0], hi = errs[0];
    for (double e : errs) {
        REQUIRE(std::isfinite(e));
        REQUIRE(e > 1.0);            // > +100 %: locked at every order
        lo = std::min(lo, e);
        hi = std::max(hi, e);
    }
    REQUIRE((hi - lo) / lo < 0.30);  // integration-insensitive
}
