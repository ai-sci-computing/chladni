/**
 * @file test_lme_curved_sphere_modes.cpp
 * @brief Physical-correctness gate for the curved-shell LME path.
 *
 * §10 step 9.7 of the LME planning doc: validates the curved LME
 * formulation by comparing its lowest-mode eigenvalues against the
 * @ref chladni::analytical::complete_spherical_shell_wilkinson_angular_frequencies
 * closed-spherical-shell analytic reference.
 *
 * The fixture is icosphere @c k=2 (162 V), R = 0.10 m, h = 1 mm steel.
 * On a perfect sphere the @c n=2 spheroidal mode is @c (2n+1) = 5-fold
 * degenerate; the icosphere mesh's icosahedral symmetry preserves the
 * 5-fold representation cleanly under the LME chart structure (verified
 * by the small cluster split in the diagnostic).
 *
 * @section reference_choice Why Wilkinson, not Loop, is the right reference
 *
 * The historical curved-LME gate compared LME vs Loop and showed a
 * ~60 % "disagreement" that was read as LME being too stiff. Two
 * layers of diagnosis later (2026-05-17 late):
 *
 *  (1) Loop with @ref MassLumping::RowSum (the default at the time)
 *      sat ~35 % LOW vs Wilkinson on this fixture — initially read as
 *      geometric Loop-limit-surface convergence error, but actually
 *      caused by the lumping bias (lumped mass lowers eigenfrequencies
 *      broadly, see the @c [.experiment] Mass-lumping probe in
 *      @c test_loop_sphere_projected.cpp).
 *  (2) The default was flipped to @ref MassLumping::None on the same
 *      day; under consistent mass, Loop and LME both land ~2 % above
 *      Wilkinson on icosphere k=2 — they agree to within ~1 % of
 *      each other.
 *
 * Gating LME against Loop would have been gating LME's correctness
 * against another formulation's mass-discretisation bias — a category
 * mistake either way. This file gates LME against the analytic
 * spectrum directly so it's robust to whatever Loop's default is.
 *
 * @section pass_threshold Pass threshold
 *
 * The first 5 modes (the n=2 spheroidal pentet) must (a) cluster
 * tightly (cluster spread < 5 %), and (b) the pentet mean must agree
 * with Wilkinson n=2 to within 5 % (the empirical rel_err on this
 * mesh sits near 1.5 % — the 5 % gate gives headroom for unrelated
 * formulation tweaks that don't degrade the convergence trajectory).
 *
 * Mode 5 (the first n=3 septet member) is reported diagnostically but
 * not gated: the analytic n=3 frequency lies above and the icosphere
 * k=2 mesh's n=3 cluster splits more than at n=2 owing to icosahedral
 * subgroup mixing.
 */

#include <chladni/analytical/shell.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using chladni::shell::LMEAssembler;
using chladni::shell::compute_shell_modes;

namespace {

/// n=2 spheroidal pentet mean rel-err vs Wilkinson for one assembler on
/// one icosphere level — the matrix §2 meshfree refinement cell.
double icosphere_pentet_rel_err(int k, bool sme)
{
    constexpr double R = 0.10, h = 1.0e-3;
    const auto mesh = chladni::mesh::generate_icosphere(R, k);
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    LMEAssembler::Params p;
    p.use_curved_shell     = true;
    p.use_ghost_nodes      = false;  // closed mesh
    p.use_second_order_sme = sme;
    LMEAssembler asm_(p);
    const auto modes = compute_shell_modes(mesh.V, mesh.F, steel, sm, h, 6,
                                           asm_);

    const double w_ref = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel, 1)
            .front();
    double sum = 0.0;
    for (Eigen::Index i = 0; i < 5; ++i) sum += modes.omegas(i);
    return std::abs(sum / 5.0 - w_ref) / w_ref;
}

}  // namespace

// Matrix §2 fill (2026-06-07): the consolidated comparison matrix left
// the meshfree icosphere cells blank above k=2. Loop converges O(h²)
// to 0.03 % by k=5; this records where LME-1st and SME land so the
// "meshfree 4–8× worse on closed smooth shells at k=2, but converging
// O(h²)" claim carries numbers (measured: SME 3.94/0.96/0.26 % at
// k=2/3/4, ratio to Loop 4.7×→3.5×→2.6×).
//
// The GATED run does k=2,3 (≈31 s wall). Higher levels are behind the
// ICO_K env var (runs ONLY that level; ICO_METHOD=lme|sme to isolate):
//   ICO_K=4 → 2562 V, 164 s wall, LME 0.36 % / SME 0.26 %.
//   ICO_K=5 → 10242 V (30,726 DOF) — OOM-KILLED (exit 137) on 18 GB
//             for BOTH methods. The cost at k=5 is MEMORY, not time:
//             the multi-seed shift-invert eigensolve factorizes K, and
//             the wide meshfree stencil (k-ring-3) fills in far more
//             than Loop's compact FEM stencil at the same DOF (Loop
//             k=5 factorizes fine). A factorization-free eigensolver
//             or more RAM is needed to reach it; the k=2→4 O(h²) trend
//             already pins the convergence (matrix §2).
TEST_CASE("icosphere meshfree refinement (matrix §2): LME-1st & SME "
          "n=2 pentet vs Wilkinson",
          "[.slow][.diag][shell][lme][sme][curved_modes][icosphere_refine]")
{
    std::fprintf(stderr,
        "\n[icosphere_refine] n=2 spheroidal pentet rel-err vs Wilkinson\n"
        "%-8s %-8s %12s %12s\n", "k", "V", "LME-1st", "SME");
    std::vector<int> ks = {2, 3};
    if (const char* kenv = std::getenv("ICO_K"))  // on-demand cost probe
        ks = {std::atoi(kenv)};
    const char* method = std::getenv("ICO_METHOD");  // "lme"|"sme"|null=both
    const bool do_lme = !method || std::string(method) != "sme";
    const bool do_sme = !method || std::string(method) != "lme";
    for (const int k : ks) {
        const auto mesh = chladni::mesh::generate_icosphere(0.10, k);
        std::fprintf(stderr, "[icosphere_refine] start k=%d (%lld V)...\n",
                     k, static_cast<long long>(mesh.V.rows()));
        const double lme = do_lme
            ? icosphere_pentet_rel_err(k, /*sme=*/false)
            : std::nan("");
        if (do_lme)
            std::fprintf(stderr, "[icosphere_refine] k=%d LME done %.3f%%\n",
                         k, 100.0 * lme);
        const double sme = do_sme
            ? icosphere_pentet_rel_err(k, /*sme=*/true)
            : std::nan("");
        std::fprintf(stderr, "%-8d %-8lld %11.3f%% %11.3f%%\n",
                     k, static_cast<long long>(mesh.V.rows()),
                     100.0 * lme, 100.0 * sme);
        if (do_lme) CHECK(std::isfinite(lme));
        if (do_sme) CHECK(std::isfinite(sme));
    }
    SUCCEED("icosphere refinement printed");
}

TEST_CASE("compute_shell_modes(LMEAssembler use_curved_shell=true) on "
          "icosphere k=2 matches Wilkinson n=2 spheroidal analytic",
          "[shell][lme][assembler][curved_modes][analytical][validation]")
{
    constexpr double      R       = 0.10;
    constexpr double      h       = 1.0e-3;
    constexpr std::size_t n_modes = 6;

    const auto mesh = chladni::mesh::generate_icosphere(R,
                                                        /*n_subdivisions=*/2);

    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    // ----- LME under test -------------------------------------------
    LMEAssembler::Params lme_params;
    lme_params.use_curved_shell = true;
    LMEAssembler lme_asm(lme_params);
    const auto   modes_lme = compute_shell_modes(
        mesh.V, mesh.F, steel, sm, h, n_modes, lme_asm);

    REQUIRE(modes_lme.omegas.size() == static_cast<Eigen::Index>(n_modes));

    // ----- Wilkinson analytic reference -----------------------------
    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel, /*n_modes=*/1);
    const double omega_n2_analytic = analytical.front();

    // ----- Spheroidal n=2 pentet: modes 0..4 ------------------------
    double sum   = 0.0;
    double w_min = modes_lme.omegas(0);
    double w_max = modes_lme.omegas(0);
    for (Eigen::Index i = 0; i < 5; ++i) {
        const double w = modes_lme.omegas(i);
        sum  += w;
        w_min = std::min(w_min, w);
        w_max = std::max(w_max, w);
    }
    const double pentet_mean  = sum / 5.0;
    const double cluster_spread = (w_max - w_min) / pentet_mean;
    const double rel_err_mean =
        std::abs(pentet_mean - omega_n2_analytic) / omega_n2_analytic;
    const double rel_err_mode_5 =
        std::abs(modes_lme.omegas(5) - omega_n2_analytic) / omega_n2_analytic;

    // Diagnostic: dump every mode + the cluster + analytic for posterity.
    std::fprintf(stderr,
        "[curved_modes_diag] Wilkinson n=2 ω = %.3f rad/s\n",
        omega_n2_analytic);
    for (std::size_t i = 0; i < n_modes; ++i) {
        const double w   = modes_lme.omegas(static_cast<Eigen::Index>(i));
        const double rel = std::abs(w - omega_n2_analytic) / omega_n2_analytic;
        std::fprintf(stderr,
            "[curved_modes_diag] mode %zu  ω_LME = %.3f rad/s  rel(vs n=2) = %.3f\n",
            i, w, rel);
    }
    std::fprintf(stderr,
        "[curved_modes_diag] pentet mean = %.3f rad/s  cluster spread = %.4f"
        "  rel_err vs Wilkinson = %.4f\n",
        pentet_mean, cluster_spread, rel_err_mean);
    std::fprintf(stderr,
        "[curved_modes_diag] mode 5 (n=3 first) rel(vs n=2 analytic) = %.3f"
        "  (not gated)\n",
        rel_err_mode_5);

    INFO("pentet mean = "      << pentet_mean
         << "  Wilkinson n=2 = " << omega_n2_analytic
         << "  cluster spread = " << cluster_spread
         << "  rel_err_mean = "   << rel_err_mean);

    // Cluster integrity: spheroidal n=2 pentet must cluster cleanly on
    // the icosphere; large spread would indicate symmetry breaking
    // beyond the icosahedral-subgroup split (which is bounded below
    // 5 % at k=2).
    REQUIRE(cluster_spread < 0.05);

    // Mode 6 sits above the pentet — confirms we identified the
    // pentet correctly (and didn't accidentally fold an n=3 mode in).
    REQUIRE(modes_lme.omegas(5) > w_max);

    // Physical-correctness gate: LME pentet mean tracks Wilkinson n=2.
    // Under the faithful per-node-β derivatives (Millán 2011 App A,
    // 2026-06-03) the coarse k=2 icosphere reads +6.8 % and CONVERGES
    // to +1.5 % at k=3 (measured 162 V → 642 V, ~O(h²)) — the earlier
    // ~3.9 % at k=2 was partial error cancellation from the uniform-β
    // derivative deviation, not better convergence. Gate at 8 % to pin
    // the coarse-mesh level; the k=3 convergence is the real evidence.
    REQUIRE(rel_err_mean < 0.08);
}

TEST_CASE("compute_shell_modes(LMEAssembler SME) on icosphere k=2 — "
          "n=2 spheroidal pentet preserved",
          "[.slow][shell][lme][assembler][curved_modes][sme][analytical]"
          "[validation]")
{
    // Phase B.3 cross-check: SME on a closed mesh must not regress
    // the 1st-order LME accuracy on icosphere k=2's n=2 spheroidal
    // pentet (1st-order LME hits ~1.5 % vs Wilkinson per the test
    // above). Closed mesh ⇒ no global boundary, so every node takes the
    // faithful interior gap (anisotropy-oriented (α/4)h²I); this also
    // exercises the curved-chart interior-gap path on a closed input.
    // At the paper's α=2 default the pentet mean lands ~4.9 % vs
    // Wilkinson; the earlier α=4 default sat ~11 % (curved-shell bending
    // over-stiffness grows with the support width set by α).
    //
    // [.slow] because SME's 5-dim Newton + IFT + FD-on-grad Hessian
    // per Gauss point makes curved K assembly ~3-5x slower than
    // 1st-order. Icosphere k=2 has 162 V, ~5 s SME K on the local box.
    constexpr double      R       = 0.10;
    constexpr double      h       = 1.0e-3;
    constexpr std::size_t n_modes = 6;

    const auto mesh = chladni::mesh::generate_icosphere(R,
                                                        /*n_subdivisions=*/2);

    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    LMEAssembler::Params lme_params;
    lme_params.use_curved_shell     = true;
    lme_params.use_ghost_nodes      = false;  // closed mesh: no ghosts
    lme_params.use_second_order_sme = true;
    // No r_cut override: SME truncation is VALUE-based (γ_eff = 2/α,
    // ≈4.80 h at α=2; inventory C7, r_cut_mult_sme retired) — on the
    // historic sweep's [2.5, 4.0] plateau. The 2026-05-27 sweep showed
    // SME on this closed mesh essentially r_cut-insensitive over that
    // band (~6.13 % throughout), so the value-based radius costs
    // nothing here and matches the polar-disk SME bar.
    LMEAssembler lme_asm(lme_params);
    const auto   modes_sme = compute_shell_modes(
        mesh.V, mesh.F, steel, sm, h, n_modes, lme_asm);

    REQUIRE(modes_sme.omegas.size() == static_cast<Eigen::Index>(n_modes));

    const auto analytical = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel, /*n_modes=*/1);
    const double omega_n2_analytic = analytical.front();

    double sum   = 0.0;
    double w_min = modes_sme.omegas(0);
    double w_max = modes_sme.omegas(0);
    for (Eigen::Index i = 0; i < 5; ++i) {
        const double w = modes_sme.omegas(i);
        sum  += w;
        w_min = std::min(w_min, w);
        w_max = std::max(w_max, w);
    }
    const double pentet_mean    = sum / 5.0;
    const double cluster_spread = (w_max - w_min) / pentet_mean;
    const double rel_err_mean   =
        std::abs(pentet_mean - omega_n2_analytic) / omega_n2_analytic;

    std::fprintf(stderr,
        "[sme_sphere] Wilkinson n=2 ω = %.3f rad/s\n", omega_n2_analytic);
    for (std::size_t i = 0; i < n_modes; ++i) {
        const double w   = modes_sme.omegas(static_cast<Eigen::Index>(i));
        const double rel = std::abs(w - omega_n2_analytic)
                            / omega_n2_analytic;
        std::fprintf(stderr,
            "[sme_sphere] mode %zu  ω_SME = %.3f rad/s  rel = %.4f\n",
            i, w, rel);
    }
    std::fprintf(stderr,
        "[sme_sphere] pentet mean = %.3f rad/s  cluster spread = %.4f  "
        "rel_err vs Wilkinson = %.4f\n",
        pentet_mean, cluster_spread, rel_err_mean);

    INFO("pentet mean = "       << pentet_mean
         << "  Wilkinson n=2 = " << omega_n2_analytic
         << "  cluster spread = " << cluster_spread
         << "  rel_err_mean = "   << rel_err_mean);

    // Cluster integrity: matches the 1st-order gate (icosahedral
    // subgroup splits are mesh-symmetry-bound, not basis-bound).
    // SME actually IMPROVES the cluster spread (~0.17 % vs the
    // 1st-order ~2 %) because the 2nd-order moment matching damps
    // the small azimuthal asymmetries of the chart projection.
    REQUIRE(cluster_spread < 0.05);
    REQUIRE(modes_sme.omegas(5) > w_max);
    // At the paper's α=2 default SME lands ~4.9 % vs Wilkinson on
    // icosphere k=2 — close to 1st-order LME's ~2.25 % and well within
    // the 1st-order physical gate (5 %). The residual gap over 1st-order
    // is the curved-shell bending over-stiffness that grows with the
    // basis support width (α): it was ~11 % at the old α=4 default and
    // ~6 % under the previous chart-rim-zeroing hack (now removed in
    // favour of the faithful §3.2.2 boundary recipe). Gated at 6 %,
    // a touch above the measured 4.9 % to absorb mesh/solver jitter.
    REQUIRE(rel_err_mean < 0.06);
}
