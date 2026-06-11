/**
 * @file test_lme_free_edge_circular_plate.cpp
 * @brief LME FEM vs Leissa Table 2.5 — free-edge circular plate (the
 *        eponymous Chladni fixture).
 *
 * Port of the Loop fixture
 * @ref tests/shell/test_modes_vs_free_edge_circular_plate_analytic.cpp
 * onto @ref chladni::shell::LMEAssembler.
 *
 * @section setup Setup
 *  - Disk: @c generate_circular_disk(R=0.10, n_az, n_rad).
 *  - Steel: E = 200 GPa, ν = 0.33, ρ = 7850 kg/m³.
 *  - Thickness h = 1 mm (R/h = 100, thin-plate regime).
 *  - Free edge — no Dirichlet BC.
 *
 * @section ref Reference (Leissa Table 2.5, ν = 0.33)
 *
 * Lowest seven angular-frequency entries, with multiplicity from
 * cos/sin pairing:
 *  - (n=2, s=0): λ² ≈ 5.253  — doubly degenerate "saddle" mode pair.
 *  - (n=0, s=1): λ² ≈ 9.084  — "breathing" ring mode.
 *  - (n=3, s=0): λ² ≈ 12.23  — trefoil pair.
 *
 * @section bc Boundary condition handling
 *
 * Free edge: no DOF pinning at the rim. The Kirchhoff variational
 * form has the boundary moment / shear absorbed into the bulk
 * integral (natural BC), so the LME assembler's standard output is
 * already the correct stiffness.
 *
 * The LME assembler ships three identical scalar blocks tiled onto
 * the 3·n_V layout (assembler-side bookkeeping). For the Kirchhoff
 * plate we only want the **z-component spectrum**: in-plane motion
 * is unphysical at this formulation order. We extract the z-DOFs via
 * an explicit selection matrix and solve a scalar
 * @f$ n_V \times n_V @f$ generalized eigenproblem (sparse Spectra
 * shift-invert, via @ref tests/shell/lme_test_helpers.hpp).
 *
 * The z-DOF spectrum has three rigid-body modes (constants + the two
 * linear motions; see the rigid-body-kernel test in
 * @c test_lme_assembler.cpp). The leading three eigenvalues are
 * therefore numerically near zero and are skipped before comparison
 * to Leissa.
 *
 * @section accuracy Accuracy — sub-1 % post derivative fix
 *
 * Migrated 2026-05-29 to the CURVED+GHOST path through @ref
 * compute_shell_modes — the formulation the GUI's LME tab runs — with the
 * paper's value-based neighbour cutoff (Params::tol_lme). Measured
 * 2026-06-04 at defaults on this fixture (32 × 8):
 *   (n=2, s=0)  +0.75 %
 *   (n=0, s=1)  +0.77 %
 *   (n=3, s=0)  +0.41 %
 * HISTORY: this header used to claim a "genuine ~8 % 1st-order LME
 * free-edge floor" (boundary moments/shears beyond a degree-1 basis,
 * curable only by 2nd-order SME). That was FALSIFIED 2026-06-03: the
 * closed-form ∇p/∇²p used the Arroyo-Ortiz 2006 UNIFORM-β special case
 * while the assembler passes per-node β_a = γ/h_a², so every assembled
 * derivative on h_a-modulated node sets was wrong by O(β-spread). With
 * the faithful Millán 2011 Appendix A per-node-β formulas (commit
 * `edaf0e1`) the fixture is sub-1 % and the [.slow] B.2 SME bar below
 * reads an equivalent +0.73 % — 1st-order LME and SME now tie on
 * bending-dominated fixtures; SME's remaining edge is membrane-dominated
 * curved shells (free-free cylinder +46 % vs LME +274 %).
 *
 * @section refinement Refinement target
 *
 * Polar-disk meshes are biased: the centre-vertex valence equals
 * @p n_az, so a coarse-azimuthal mesh has a very high-valence
 * extraordinary vertex there. For the LME basis this doesn't matter
 * (no special handling for extraordinary vertices — the basis is
 * meshfree); the disk geometry just acts as a quadrature domain.
 * 32 × 8 (257 V, 480 F) gives a fast-running fixture (~1.2 s) in the
 * same resolution band as the Loop counterpart.
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include "lme_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <chladni/shell.hpp>

using chladni::shell::LMEAssembler;
using chladni::shell::ShellMaterial;
using chladni::tests::lme::solve_lme_z_modes;

namespace {

chladni::IsotropicMaterial steel_033()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.33,
            .density        = 7850.0};
}

struct LMEDiskRun {
    Eigen::Index n_v;
    std::vector<double> omegas_hz;  // ascending, post-rigid-skip
};

/// Solve the free-edge disk via the CURVED+GHOST path through the app's
/// actual modal pipeline (@ref compute_shell_modes, which rigid-filters and
/// handles the ghost-extended spectrum), returning the lowest @p n_modes
/// physical angular frequencies (Hz). Migrated 2026-05-29 from the flat
/// z-block path so the fixture mirrors what the GUI's LME tab runs.
LMEDiskRun run_lme_disk(int n_az, int n_rad, std::size_t n_modes)
{
    constexpr double R   = 0.10;
    constexpr double h   = 1.0e-3;
    const auto mat       = steel_033();
    const auto mesh      = chladni::mesh::generate_circular_disk(R, n_az, n_rad);

    LMEAssembler::Params p;          // default: curved + ghost (app path)
    p.newton_tol = 1e-13;
    p.newton_max_iters = 60;
    LMEAssembler asm_(p);

    const auto sm = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, mat, sm, h, n_modes, asm_);

    LMEDiskRun out;
    out.n_v = mesh.V.rows();
    out.omegas_hz.reserve(n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (std::size_t k = 0; k < n_modes; ++k) {
        out.omegas_hz.push_back(modes.omegas(static_cast<Eigen::Index>(k)) / two_pi);
    }
    return out;
}

}  // namespace

TEST_CASE("LME FEM vs Leissa: free-edge circular plate — Chladni fixture",
          "[lme][assembler][modes][plate][circular][analytic][validation]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes_physical = 5;
    constexpr std::size_t n_rigid_z = 3;  // const + linear x + linear y

    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, steel_033(), n_modes_physical);
    REQUIRE(analytic.size() == n_modes_physical);

    // 32 × 8 disk: 257 V, 480 F. Same resolution band as the Loop
    // counterpart for direct A/B comparison.
    (void)n_rigid_z;  // compute_shell_modes rigid-filters internally
    const auto r = run_lme_disk(/*n_az=*/32, /*n_rad=*/8, n_modes_physical);

    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    std::cout << "[LME free-edge circular plate R=" << R << " h=" << h
              << "  V=" << r.n_v << "]\n  Leissa (Hz):";
    for (double w : analytic) std::cout << " " << w / two_pi;
    std::cout << "\n  LME    (Hz):";
    for (double w : r.omegas_hz) std::cout << " " << w;
    std::cout << '\n';

    auto rel = [](double f, double f_ref) {
        return std::abs(f - f_ref) / f_ref;
    };

    // CURVED+GHOST path (the app's LME path) at 32 × 8, measured
    // 2026-06-04 under the per-node-β derivatives (`edaf0e1`):
    //   (n=2, s=0)  LME +0.75 %
    //   (n=0, s=1)  LME +0.77 %
    //   (n=3, s=0)  LME +0.41 %
    // Gated at 2 % for mesh-generator / run-to-run headroom; a breach
    // means the derivative formulas or the free-edge machinery
    // regressed. (The pre-2026-06-03 revision gated 10 % around the
    // "~8 % 1st-order floor" — see the header; that floor was the
    // uniform-β derivative bug.)

    const double f_n2 = 0.5 * (r.omegas_hz[0] + r.omegas_hz[1]);
    const double f_n2_ref = analytic[0] / two_pi;
    INFO("(n=2, s=0) LME mean = " << f_n2 << " Hz, Leissa = " << f_n2_ref
         << " Hz, rel_err = " << rel(f_n2, f_n2_ref));
    REQUIRE(rel(f_n2, f_n2_ref) < 0.02);

    const double f_n0 = r.omegas_hz[2];
    const double f_n0_ref = analytic[2] / two_pi;
    INFO("(n=0, s=1) LME = " << f_n0 << " Hz, Leissa = " << f_n0_ref
         << " Hz, rel_err = " << rel(f_n0, f_n0_ref));
    REQUIRE(rel(f_n0, f_n0_ref) < 0.02);

    const double f_n3 = 0.5 * (r.omegas_hz[3] + r.omegas_hz[4]);
    const double f_n3_ref = analytic[3] / two_pi;
    INFO("(n=3, s=0) LME mean = " << f_n3 << " Hz, Leissa = " << f_n3_ref
         << " Hz, rel_err = " << rel(f_n3, f_n3_ref));
    REQUIRE(rel(f_n3, f_n3_ref) < 0.02);

    // Cluster splits — keep them sub-2% to catch a runaway symmetry
    // break (e.g. bug in node-spacing computation breaks azimuthal
    // homogeneity of the basis). These are structural and shouldn't
    // grow as the basis-order is improved.
    REQUIRE(std::abs(r.omegas_hz[1] - r.omegas_hz[0]) / f_n2 < 0.02);
    REQUIRE(std::abs(r.omegas_hz[4] - r.omegas_hz[3]) / f_n3 < 0.02);
}

TEST_CASE("Timing budget — LME on a 129-V polar disk (perf-iteration fixture)",
          "[lme][circular][timing][experiment][.experiment][perf]")
{
    // Iteration vehicle for perf work. 32x4 polar disk = 1 + 32*4 =
    // 129 V is small enough that LME-flat K completes in single-digit
    // seconds and warm-start changes show a clean signal. Compared to
    // the 577-V probe below, this fixture has ~94x less work — the
    // big probe is too slow to run repeatedly while iterating.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 4;
    constexpr std::size_t n_modes = 12;

    const auto mat   = steel_033();
    const auto sm    = chladni::shell::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    const auto mesh  = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[disk " << n_az << "x" << n_rad << ", V=" << n_v
              << ", 3V=" << (3*n_v) << ", n_modes=" << n_modes << "]\n";

    auto t_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto t_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Loop baseline.
    {
        chladni::shell::LoopAssembler::Params p;
        p.use_stam = false;
        chladni::shell::LoopAssembler asm_(p);
        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        std::cout << "  Loop (L.3.4):  K=" << std::setw(8) << t_ms(t0, t1)
                  << "ms  M=" << std::setw(8) << t_ms(t1, t2) << "ms\n";
    }

    // LME flat path.
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = false;
        chladni::shell::LMEAssembler asm_(p);
        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        std::cout << "  LME (flat):    K=" << std::setw(8) << t_ms(t0, t1)
                  << "ms  M=" << std::setw(8) << t_ms(t1, t2) << "ms\n";
    }

    // LME curved path.
    try {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        chladni::shell::LMEAssembler asm_(p);
        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        std::cout << "  LME (curved):  K=" << std::setw(8) << t_ms(t0, t1)
                  << "ms  M=" << std::setw(8) << t_ms(t1, t2) << "ms\n";
    } catch (const std::exception& e) {
        std::cout << "  LME (curved):  FAILED — " << e.what() << '\n';
    }

    // SME curved+ghost path. SME's truncation is VALUE-based
    // (γ_eff = 2/α ⇒ ≈4.80 h at α=2; inventory C7, r_cut_mult_sme
    // retired) — the wide active set it needs (1.4 was too tight for
    // SME and diverged).
    try {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell      = true;
        p.use_ghost_nodes       = true;
        p.use_second_order_sme  = true;
        chladni::shell::LMEAssembler asm_(p);
        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        std::cout << "  SME (curved):  K=" << std::setw(8) << t_ms(t0, t1)
                  << "ms  M=" << std::setw(8) << t_ms(t1, t2) << "ms\n";
    } catch (const std::exception& e) {
        std::cout << "  SME (curved):  FAILED — " << e.what() << '\n';
    }
    std::cout << std::defaultfloat;
}

TEST_CASE("Timing breakdown — LME on a 2.3k-V polar disk (GUI flow)",
          "[lme][circular][timing][experiment][.experiment]")
{
    // Replicate the strike_gui flow on the disk size the user reported
    // slow: 144 x 16 polar disk = 1 + 144*16 = 2305 V. Time the K
    // assembly, M assembly, and the multi-seed eigensolve separately so
    // we know which dominates. Comparison to Loop on the same mesh
    // tells us whether LME's bottleneck is the eigensolve (shared with
    // Loop) or LME-specific assembly.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    // 72 x 8 → 577 V. Small enough that LME-curved K assembly
    // completes in seconds (the 144x16 case the user actually hit
    // takes >25 min on a single core — the curved-shell wPCA +
    // Shepard + Newton-per-Gauss-point machinery dominates).
    constexpr int n_az  = 72;
    constexpr int n_rad = 8;
    constexpr std::size_t n_modes = 24;

    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    const auto mesh = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[disk " << n_az << "x" << n_rad << ", V=" << n_v
              << ", 3V=" << (3*n_v) << ", n_modes=" << n_modes << "]\n";

    auto t_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto t_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Loop path first (fast — establishes the eigensolve baseline cost).
    {
        chladni::shell::LoopAssembler::Params p;
        p.use_stam = false;
        chladni::shell::LoopAssembler asm_(p);

        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        auto modes = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h_thk, n_modes, asm_);
        const auto t3 = t_now();
        REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));

        const double dt_K   = t_ms(t0, t1);
        const double dt_M   = t_ms(t1, t2);
        const double dt_full = t_ms(t2, t3);
        std::cout << "  Loop (L.3.4):  K=" << std::setw(8) << dt_K
                  << "ms  M=" << std::setw(8) << dt_M
                  << "ms  full(K+M+eig)=" << std::setw(8) << dt_full
                  << "ms  → eig≈" << (dt_full - dt_K - dt_M) << "ms\n";
    }

    // LME flat path (use_curved_shell=false).
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = false;
        chladni::shell::LMEAssembler asm_(p);

        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        std::cout << "  LME (flat):    K=" << std::setw(8) << t_ms(t0, t1)
                  << "ms  M=" << std::setw(8) << t_ms(t1, t2) << "ms\n";
    }

    // LME curved path LAST (slowest). The curved-shell path used to
    // crash here: per-Gauss-point Newton diverged at marginal Shepard
    // patches whose k-ring=5 BFS hit the high-valence (valence=72)
    // disk-centre vertex and exploded the chart to 137-196 nodes; the
    // wPCA projection of that degenerate point cloud has chart
    // geometries where the in-chart dual has no finite minimiser. The
    // catch-and-renormalise safety net inside assemble_K_curved_bending
    // / assemble_M_curved now drops the offending patch from the
    // Shepard PoU and renormalises the surviving weights, so the curved
    // path completes. It remains very slow at this size — the actual
    // O(10³x) gap vs Loop is the real perf problem (separate followup).
    // Try/catch retained as defence in depth: if a future code change
    // breaks curved-shell assembly the probe still records the Loop +
    // LME-flat timings and reports the curved failure rather than
    // aborting the whole test case.
    try {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        chladni::shell::LMEAssembler asm_(p);

        const auto t0 = t_now();
        auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto t1 = t_now();
        auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);
        const auto t2 = t_now();
        auto modes = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h_thk, n_modes, asm_);
        const auto t3 = t_now();
        REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));

        const double dt_K   = t_ms(t0, t1);
        const double dt_M   = t_ms(t1, t2);
        const double dt_full = t_ms(t2, t3);
        std::cout << "  LME (curved):  K=" << std::setw(8) << dt_K
                  << "ms  M=" << std::setw(8) << dt_M
                  << "ms  full(K+M+eig)=" << std::setw(8) << dt_full
                  << "ms  → eig≈" << (dt_full - dt_K - dt_M) << "ms\n";
    } catch (const std::exception& e) {
        std::cout << "  LME (curved):  FAILED — " << e.what() << '\n';
    }
    std::cout << std::defaultfloat;
}

TEST_CASE("LME curved+ghost on 32x8 polar disk vs Leissa — shipped-defaults gate",
          "[lme][circular][curved][ghost][validation]")
{
    // 1st-order LME curved+ghost at the SHIPPED defaults on the 32x8
    // polar-disk free-edge Leissa fixture.
    //
    // HISTORY (corrected 2026-05-29): an earlier revision pinned
    // r_cut_mult_curved=1.4, which gave ~0.6% here and was guarded as a
    // "win". That was an OVER-TUNING ARTIFACT: the narrow fixed radius
    // accidentally cancelled errors on this one fixture while STARVING the
    // bending Hessian on structured/curved meshes (~0.64x analytic bending
    // energy; Scordelis-Lo ~22% too soft) — see
    // test_static_obstacle_course.cpp. The curved path now uses the paper's
    // VALUE-based neighbour cutoff (Millán 2011 Eq. 2, Params::tol_lme),
    // which is mesh-adaptive and gives correct bending everywhere.
    //
    // HISTORY (corrected 2026-06-03): the "~7.9% genuine 1st-order LME
    // free-edge floor" this test used to pin was NOT a floor — it was
    // the uniform-β derivative deviation: the closed-form ∇p/∇²p used
    // the Arroyo-Ortiz 2006 eq. 44 UNIFORM-β special case while the
    // assembler passes per-node β_a = γ/h_a² (the polar disk's
    // rim/centre h_a variation made every assembled derivative wrong
    // by O(β-spread)). With the faithful Millán 2011 Appendix A
    // per-node-β formulas the same fixture reads ~+0.75% on the (n=2)
    // doublet — and modes (0,1) and (3,0) land sub-1% too. The same
    // bug was the root cause of the QuadSplit spurious-mode failure
    // (see the [quadsplit_sweep] diag below).
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto mesh = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;

    chladni::shell::LMEAssembler::Params p;
    p.use_curved_shell = true;
    p.use_ghost_nodes  = true;
    chladni::shell::LMEAssembler asm_(p);

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
    const double f_n2 = 0.5 *
        (modes.omegas(0) + modes.omegas(1)) / two_pi;
    const double rel_err =
        std::abs(f_n2 - f_n2_ref) / f_n2_ref;

    INFO("LME curved+ghost  Leissa n=2 = " << f_n2_ref
         << " Hz  LME = " << f_n2 << " Hz  rel_err = " << rel_err);
    // Measured +0.75% under the faithful per-node-β derivatives
    // (2026-06-03). Gate at 2% for mesh-generator / run-to-run
    // headroom; a regression above it means either the derivative
    // formulas or the curved free-edge machinery degraded. (The old
    // 5-9% band pinned the "1st-order floor" that turned out to be
    // the uniform-β derivative bug — see the header comment.)
    REQUIRE(rel_err < 0.02);

    // Full 7-mode pin through the (1,1) doublet (added 2026-06-06,
    // the "(1,1) column" item). Analytic ordering at ν=0.33:
    // indices 0-1 = (2,0) doublet, 2 = (0,1) singlet, 3-4 = (3,0)
    // doublet, 5-6 = (1,1) doublet (the next root, (4,0), lies ~5%
    // above (1,1) — far beyond the sub-1% errors gated here, so
    // index-matching is unambiguous). Measured at the shipped
    // defaults ([disk_table] 2026-06-06): (0,1) +0.78%, (3,0)
    // +0.41%, (1,1) +0.03%. Same 2% bar as the n=2 gate.
    struct Cluster { const char* label; int i0; int i1; };
    for (const Cluster c : {Cluster{"(0,1)", 2, 2},
                            Cluster{"(3,0)", 3, 4},
                            Cluster{"(1,1)", 5, 6}}) {
        const double f_ref = 0.5
            * (analytic[static_cast<std::size_t>(c.i0)]
               + analytic[static_cast<std::size_t>(c.i1)]) / two_pi;
        const double f = 0.5
            * (modes.omegas(c.i0) + modes.omegas(c.i1)) / two_pi;
        const double err = std::abs(f - f_ref) / f_ref;
        INFO("LME curved+ghost  Leissa " << c.label << " = " << f_ref
             << " Hz  LME = " << f << " Hz  rel_err = " << err);
        REQUIRE(err < 0.02);
    }
}

TEST_CASE("LME curved+ghost on symmetric QuadSplit 32x8 disks vs Leissa — "
          "regression gates",
          "[lme][circular][curved][ghost][quadsplit][validation]")
{
    // REGRESSION GATES for the 2026-06-03 QuadSplit-rim bug. The
    // symmetric QuadSplit meshes (Checkerboard / UnionJack) used to
    // break the 1st-order LME ghost path catastrophically (−31.7 % /
    // −19.5 % with doublet splits 0.98 / 0.36): a spurious m=0
    // sublattice-zigzag mode BELOW the physical spectrum, root-caused
    // to the uniform-β closed-form derivatives applied to per-node
    // β_a = γ/h_a² (strongly modulated between the UJ/CB sublattices).
    // With the faithful Millán 2011 App A per-node-β derivatives all
    // three splits land sub-1 % with cleanly degenerate doublets:
    // Consistent +0.75 %, Checkerboard +0.43 %, UnionJack +0.50 %
    // (splits ≤ 1.4e-4).
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;

    using chladni::mesh::QuadSplit;
    for (const QuadSplit split :
         {QuadSplit::Checkerboard, QuadSplit::UnionJack}) {
        const auto mesh =
            chladni::mesh::generate_circular_disk(R, 32, 8, split);
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = true;
        chladni::shell::LMEAssembler asm_(p);
        const auto modes = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
        const double f0 = modes.omegas(0) / two_pi;
        const double f1 = modes.omegas(1) / two_pi;
        const double f_n2     = 0.5 * (f0 + f1);
        const double rel_err  = std::abs(f_n2 - f_n2_ref) / f_n2_ref;
        const double split_n2 = std::abs(f1 - f0) / f_n2;
        INFO("split=" << static_cast<int>(split)
             << "  f(n=2)=" << f_n2 << " Hz  rel_err=" << rel_err
             << "  doublet_split=" << split_n2);
        // Sub-2% accuracy + a genuinely degenerate doublet (the
        // spurious-mode failure showed up as splits of 0.36-0.98, so
        // 1e-2 cleanly separates regression from noise).
        REQUIRE(rel_err < 0.02);
        REQUIRE(split_n2 < 0.01);
    }
}

TEST_CASE("LME curved+ghost QuadSplit sweep on 32x8 polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][quadsplit_sweep]")
{
    // CHARACTERIZATION (2026-06-03 handoff, RESOLVED same day): the
    // symmetric QuadSplit meshes (Checkerboard / UnionJack) broke the
    // 1st-order LME ghost path — UnionJack −19.5 % (doublet split
    // 0.36), Checkerboard −31.7 % (split 0.98) — while SME on the SAME
    // meshes was fine. The investigation chain recorded in this diag
    // (ownership/Newton drops clean → spurious m=0 sublattice-zigzag
    // mode below the analytic spectrum → quadrature order + chart
    // extent + ghost row geometry all innocent → FLAT global-basis
    // contrast clean → conforming-z reference pencil reproduces the
    // spurious mode with energy ratio 1.03 → quadrature-cell
    // refinement converged ⇒ the integrand's DERIVATIVES were wrong)
    // led to the root cause: uniform-β closed-form ∇p/∇²p applied to
    // per-node β_a (fixed faithfully per Millán 2011 App A; see the
    // [nonuniform_beta] FD test). This sweep prints rel_err + doublet
    // split for all three splits plus the diagnostic contrasts.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;
    std::cout << "[quadsplit_sweep] Leissa n=2 ref = " << f_n2_ref
              << " Hz\n";

    struct Case {
        const char*              name;
        chladni::mesh::QuadSplit split;
        bool                     ghosts;
        int                      n_quad;
    };
    const Case cases[] = {
        {"Consistent  ghost-ON ", chladni::mesh::QuadSplit::Consistent,   true,  12},
        {"Checkerboard ghost-ON", chladni::mesh::QuadSplit::Checkerboard, true,  12},
        {"UnionJack   ghost-ON ", chladni::mesh::QuadSplit::UnionJack,    true,  12},
        {"UnionJack   ghost-OFF", chladni::mesh::QuadSplit::UnionJack,    false, 12},
        // Quadrature-sensitivity probes: a spurious eigenvalue BELOW
        // the analytic spectrum on a conforming basis can only come
        // from energy under-integration; if real, it must move
        // strongly with the rule order. MEASURED: insensitive (102.7
        // vs 103.5 Hz) — rule order is innocent.
        {"UnionJack   ghost-ON 7pt ", chladni::mesh::QuadSplit::UnionJack, true, 7},
        {"Checkerboard ghost-ON 7pt", chladni::mesh::QuadSplit::Checkerboard, true, 7},
        // FLAT-path discriminator (diagnostic contrast ONLY — the
        // production path is and stays the curved one): one GLOBAL 3D
        // LME basis, conforming Galerkin → variational upper bound
        // holds. MEASURED: no spurious mode (UJ doublet split 3.5e-10)
        // ⇒ the energy deficit lives in the curved path's per-patch
        // chart/PoU machinery, NOT the two-sublattice node set.
        {"UnionJack   ghost-ON FLAT", chladni::mesh::QuadSplit::UnionJack, true, 12},
        {"Consistent  ghost-ON FLAT", chladni::mesh::QuadSplit::Consistent, true, 12},
        // Chart-extent probe ON THE CURVED PATH: chart_tol_lme=1e-6
        // (the perf compromise that sizes the chart extent below the
        // basis support's tol_lme=1e-10) was validated on consistent
        // meshes only — test whether the UJ spurious mode is
        // chart-truncation-driven.
        {"UnionJack   ghost-ON WIDE", chladni::mesh::QuadSplit::UnionJack, true, 12},
    };

    for (const auto& c : cases) {
        const auto mesh =
            chladni::mesh::generate_circular_disk(R, 32, 8, c.split);

        const bool flat = std::string(c.name).find("FLAT") != std::string::npos;
        const bool wide = std::string(c.name).find("WIDE") != std::string::npos;
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = !flat;
        p.use_ghost_nodes  = c.ghosts;
        p.n_quadrature_per_tri = c.n_quad;
        if (wide) {
            p.chart_tol_lme   = 1e-10;   // full basis-support extent
            p.max_chart_nodes = 100000;  // disable the high-valence cap
        }
        chladni::shell::LMEAssembler asm_(p);

        try {
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
            const double f0 = modes.omegas(0) / two_pi;
            const double f1 = modes.omegas(1) / two_pi;
            const double f_n2 = 0.5 * (f0 + f1);
            const double rel_err = (f_n2 - f_n2_ref) / f_n2_ref;
            const double split   = std::abs(f1 - f0) / f_n2;
            std::cout << "[quadsplit_sweep] " << c.name
                      << "  V=" << mesh.V.rows()
                      << "  f(n=2)=" << f_n2
                      << " Hz  rel_err=" << rel_err
                      << "  doublet_split=" << split << "\n";
        } catch (const std::exception& e) {
            std::cout << "[quadsplit_sweep] " << c.name
                      << "  V=" << mesh.V.rows()
                      << "  THREW: " << e.what() << "\n";
        }
    }
    SUCCEED();
}

TEST_CASE("LME curved+ghost QuadSplit pencil conditioning on 32x8 polar disk",
          "[.diag][lme][circular][curved][ghost][quadsplit_pencil]")
{
    // Drill-down for the QuadSplit-rim bug: dense spectral analysis of
    // the assembled (K, M) pencil per split. Checks (a) M's smallest
    // eigenvalues (near-zero ⇒ nearly linearly dependent basis
    // functions, e.g. the Checkerboard rim's near-coincident ghost
    // pairs), (b) K's near-null count (must be exactly 6 rigid-body
    // modes — more ⇒ spurious mechanisms), and (c) ghost-DOF
    // coefficient participation in the lowest physical modes.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    struct Case {
        const char*              name;
        chladni::mesh::QuadSplit split;
        bool                     ghosts;
    };
    const Case cases[] = {
        {"Consistent  ", chladni::mesh::QuadSplit::Consistent,   true},
        {"Checkerboard", chladni::mesh::QuadSplit::Checkerboard, true},
        {"UnionJack   ", chladni::mesh::QuadSplit::UnionJack,    true},
        // Ghost-OFF contrast: is the spurious sublattice-zigzag mode
        // CREATED by the ghost machinery, or merely UNMASKED by it
        // (present ghost-off too, just stiffened above the locked
        // doublet)?
        {"UnionJack OFF", chladni::mesh::QuadSplit::UnionJack,   false},
    };

    for (const auto& c : cases) {
        const auto mesh =
            chladni::mesh::generate_circular_disk(R, 32, 8, c.split);
        const Eigen::Index n_v = mesh.V.rows();

        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = c.ghosts;
        chladni::shell::LMEAssembler asm_(p);

        const Eigen::SparseMatrix<double> K =
            asm_.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::SparseMatrix<double> M =
            asm_.assemble_M(mesh.V, mesh.F, mat.density * h);
        const Eigen::Index n_dof  = K.rows();
        const Eigen::Index n_gdof = n_dof - 3 * n_v;

        const Eigen::MatrixXd Kd = Eigen::MatrixXd(K);
        const Eigen::MatrixXd Md = Eigen::MatrixXd(M);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_m(
            Md, Eigen::EigenvaluesOnly);
        const Eigen::VectorXd m_ev = es_m.eigenvalues();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_k(
            Kd, Eigen::EigenvaluesOnly);
        const Eigen::VectorXd k_ev = es_k.eigenvalues();
        const double k_max = k_ev(k_ev.size() - 1);
        Eigen::Index k_null = 0;
        for (Eigen::Index i = 0; i < k_ev.size(); ++i) {
            if (std::abs(k_ev(i)) < 1e-10 * k_max) ++k_null;
        }

        std::cout << "[quadsplit_pencil] " << c.name
                  << "  n_dof=" << n_dof << " (ghost dof " << n_gdof << ")\n"
                  << "  M eigs: min=" << m_ev(0)
                  << "  2nd=" << m_ev(1)
                  << "  3rd=" << m_ev(2)
                  << "  max=" << m_ev(m_ev.size() - 1)
                  << "  cond=" << m_ev(m_ev.size() - 1) / m_ev(0) << "\n"
                  << "  K: max=" << k_max
                  << "  near-null(<1e-10*max)=" << k_null
                  << "  (expect 6)\n";

        // Generalized modes + ghost participation of the lowest 10
        // non-rigid ones (coefficient-space fractions).
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
            Kd, Md, Eigen::ComputeEigenvectors);
        const Eigen::VectorXd evals = ges.eigenvalues();
        constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
        std::cout << "  lowest 12 modes (Hz | ghost-coeff frac):";
        for (Eigen::Index i = 0; i < 12 && i < evals.size(); ++i) {
            const double lam = std::max(evals(i), 0.0);
            const double f   = std::sqrt(lam) / two_pi;
            const Eigen::VectorXd& v = ges.eigenvectors().col(i);
            const double frac =
                v.tail(n_gdof).squaredNorm() / v.squaredNorm();
            std::cout << "\n    f=" << f << "  ghost_frac=" << frac;
        }
        std::cout << "\n";

        // Shape analysis of modes 6-8 (first non-rigid trio): radial
        // localisation (coefficient mass binned by |x| in units of R)
        // and the azimuthal wavenumber spectrum of the rim ring's
        // z-coefficients — a spurious rim/ghost mechanism shows up as
        // rim+ghost-concentrated mass with a high-m (e.g. m=16)
        // signature, vs the physical n=2 doublet's m=2.
        for (Eigen::Index mi = 6; mi < 9 && mi < evals.size(); ++mi) {
            const Eigen::VectorXd& v = ges.eigenvectors().col(mi);
            double bin_inner = 0.0, bin_mid = 0.0, bin_rim = 0.0,
                   bin_ghost = v.tail(n_gdof).squaredNorm();
            std::vector<std::pair<double, double>> rim_az_z;
            for (Eigen::Index a = 0; a < n_v; ++a) {
                const double r = mesh.V.row(a).head<2>().norm() / R;
                const double m2 = v.segment(3 * a, 3).squaredNorm();
                if (r > 0.99) {
                    bin_rim += m2;
                    rim_az_z.emplace_back(
                        std::atan2(mesh.V(a, 1), mesh.V(a, 0)),
                        v(3 * a + 2));
                } else if (r > 0.6) {
                    bin_mid += m2;
                } else {
                    bin_inner += m2;
                }
            }
            const double tot = bin_inner + bin_mid + bin_rim + bin_ghost;
            // Crude azimuthal DFT of rim z-coefficients, m = 0..16.
            std::sort(rim_az_z.begin(), rim_az_z.end());
            double best_amp = 0.0;
            int    best_m   = -1;
            for (int m = 0; m <= 16; ++m) {
                double cs = 0.0, sn = 0.0;
                for (const auto& [th, z] : rim_az_z) {
                    cs += z * std::cos(m * th);
                    sn += z * std::sin(m * th);
                }
                const double amp = std::sqrt(cs * cs + sn * sn);
                if (amp > best_amp) { best_amp = amp; best_m = m; }
            }
            // Polarisation: z (bending) vs in-plane (membrane) fraction
            // over the REAL vertices — a soft in-plane mode would be a
            // membrane mechanism, not a bending mode.
            double z_mass = 0.0, xy_mass = 0.0;
            for (Eigen::Index a = 0; a < n_v; ++a) {
                z_mass  += v(3 * a + 2) * v(3 * a + 2);
                xy_mass += v.segment(3 * a, 2).squaredNorm();
            }
            const double f = std::sqrt(std::max(evals(mi), 0.0)) / two_pi;
            std::cout << "  mode[" << mi << "] f=" << f
                      << "  coeff-mass r<0.6:" << bin_inner / tot
                      << " 0.6-0.99:" << bin_mid / tot
                      << " rim:" << bin_rim / tot
                      << " ghost:" << bin_ghost / tot
                      << "  z-frac(real)=" << z_mass / (z_mass + xy_mass)
                      << "  rim-z dominant m=" << best_m << "\n";

            // FIELD-CANCELLATION test: GES returns vᵀMv = 1 (unit field
            // L² norm, M is consistent), so the coefficient norm |v|²
            // measures coefficients-per-unit-field. A genuinely soft
            // deformation has |v|² comparable to the physical modes; a
            // near-dependence direction of the extended basis (coeffs
            // cancelling in the interpolated field) shows |v|² orders
            // of magnitude larger — λ is then a 0/0 conditioning
            // artifact, not physics. Also print the (weak-δ) nodal-z
            // radial profile along θ = 0.
            std::cout << "    |coeff|^2 (v'Mv=1) = " << v.squaredNorm()
                      << "  z-coeff(theta=0 ring 1..8)=";
            for (int rr = 1; rr <= 8; ++rr) {
                // ring-rr vertex at azimuth index 0 (theta = 0).
                const Eigen::Index idx = 1 + (rr - 1) * 32;
                std::cout << " " << v(3 * idx + 2);
            }
            std::cout << "  centre=" << v(2) << "\n";
            if (c.split == chladni::mesh::QuadSplit::UnionJack) {
                // Second sublattice: quad-centre vertices along the
                // same radial line (theta = half-step) — a sublattice
                // zigzag hides from the ring-only profile.
                const Eigen::Index ctr_base = 1 + 8 * 32;
                std::cout << "    z-coeff(ctr verts, strips 1..7)=";
                for (int rr = 1; rr <= 7; ++rr) {
                    const Eigen::Index idx = ctr_base + (rr - 1) * 32;
                    std::cout << " " << v(3 * idx + 2);
                }
                std::cout << "\n";
            }

            // Energy/mass localisation: e_a = v_aᵀ(Kv)_a per 3-DOF
            // block (sums to vᵀKv = λ), same for M. Shows WHERE the
            // spurious mode's residual strain energy lives.
            // Cross-energy under the FLAT (conforming global-basis)
            // operators: same node+ghost DOF layout, so vᵀK_flat v is
            // the energy the conforming discretisation assigns to the
            // SAME coefficient vector. vᵀK_flat v >> vᵀK_curved v
            // localises a concrete energy deficit in the curved
            // assembly (not just a basis-smoothing story).
            if (mi == 6) {
                chladni::shell::LMEAssembler::Params pf = p;
                pf.use_curved_shell = false;
                chladni::shell::LMEAssembler asm_flat(pf);
                const Eigen::SparseMatrix<double> Kf =
                    asm_flat.assemble_K(mesh.V, mesh.F, sm);
                const Eigen::SparseMatrix<double> Mf =
                    asm_flat.assemble_M(mesh.V, mesh.F, mat.density * h);
                if (Kf.rows() == n_dof) {
                    const double eK_flat = v.dot(Kf * v);
                    const double eM_flat = v.dot(Mf * v);
                    const double eK_curv = v.dot(Kd * v);
                    std::cout << "    cross-energy: v'K_flat v="
                              << eK_flat
                              << "  v'K_curved v=" << eK_curv
                              << "  ratio=" << eK_flat / eK_curv
                              << "  v'M_flat v=" << eM_flat << "\n";
                } else {
                    std::cout << "    cross-energy: flat dof "
                              << Kf.rows() << " != curved dof "
                              << n_dof << " (skipped)\n";
                }
            }

            const Eigen::VectorXd Kv = Kd * v;
            const Eigen::VectorXd Mv = Md * v;
            double e_in = 0, e_mid = 0, e_rim = 0, e_gh = 0;
            double q_in = 0, q_mid = 0, q_rim = 0, q_gh = 0;
            for (Eigen::Index a = 0; a < n_dof / 3; ++a) {
                const double ea =
                    v.segment(3 * a, 3).dot(Kv.segment(3 * a, 3));
                const double qa =
                    v.segment(3 * a, 3).dot(Mv.segment(3 * a, 3));
                if (a >= n_v) { e_gh += ea; q_gh += qa; continue; }
                const double r = mesh.V.row(a).head<2>().norm() / R;
                if (r > 0.99)      { e_rim += ea; q_rim += qa; }
                else if (r > 0.6)  { e_mid += ea; q_mid += qa; }
                else               { e_in  += ea; q_in  += qa; }
            }
            const double e_tot = e_in + e_mid + e_rim + e_gh;
            std::cout << "    energy frac: r<0.6:" << e_in / e_tot
                      << " 0.6-0.99:" << e_mid / e_tot
                      << " rim:" << e_rim / e_tot
                      << " ghost:" << e_gh / e_tot
                      << "   mass frac: " << q_in << "/" << q_mid
                      << "/" << q_rim << "/" << q_gh << "\n";
        }
    }
    SUCCEED();
}

TEST_CASE("UnionJack conforming-z reference pencil (nodes+ghosts) vs curved",
          "[.diag][lme][circular][quadsplit_conforming]")
{
    // DECISIVE probe for the QuadSplit spurious-mode bug: a CONFORMING
    // scalar Kirchhoff z-pencil assembled over the SAME extended node
    // set (UJ vertices + the ghost row) with one GLOBAL planar LME
    // basis, mirroring the curved path's β = γ/h_a² and value-based
    // r_cut. A conforming Galerkin discretisation has the variational
    // upper-bound property — its spectrum CANNOT dip below the
    // analytic free-plate values. Two outputs:
    //  (1) the conforming pencil's low spectrum (must show NO ~84 Hz
    //      intruder);
    //  (2) the conforming Rayleigh quotient of the curved pencil's
    //      spurious eigenvector — if it lands near the physical band,
    //      the curved assembly concretely underprices that field, and
    //      the difference between the two assemblies IS the bug.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;

    const auto mesh = chladni::mesh::generate_circular_disk(
        R, 32, 8, chladni::mesh::QuadSplit::UnionJack);
    const Eigen::Index n_v = mesh.V.rows();

    // --- extended node set: vertices + ghost row (assembler recipe) --
    const auto bdry = chladni::shell::lme::collect_boundary_edges(mesh.F);
    const Eigen::MatrixXd ghosts =
        chladni::shell::lme::build_ghost_positions(mesh.V, mesh.F, bdry);
    const Eigen::Index G     = ghosts.rows();
    const Eigen::Index n_ext = n_v + G;

    Eigen::MatrixXd nodes2d(n_ext, 2);
    nodes2d.topRows(n_v)    = mesh.V.leftCols(2);
    nodes2d.bottomRows(G)   = ghosts.leftCols(2);

    // --- per-node h (mean incident edge length) + ghost h ------------
    Eigen::VectorXd h_sum = Eigen::VectorXd::Zero(n_v);
    Eigen::VectorXi h_cnt = Eigen::VectorXi::Zero(n_v);
    std::set<std::pair<int, int>> seen;
    for (Eigen::Index t = 0; t < mesh.F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            int i = mesh.F(t, k), j = mesh.F(t, (k + 1) % 3);
            if (i > j) std::swap(i, j);
            if (!seen.insert({i, j}).second) continue;
            const double len =
                (mesh.V.row(i) - mesh.V.row(j)).norm();
            h_sum(i) += len; h_cnt(i) += 1;
            h_sum(j) += len; h_cnt(j) += 1;
        }
    }
    Eigen::VectorXd h_a(n_ext);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        h_a(a) = h_sum(a) / std::max(1, h_cnt(a));
    }
    for (Eigen::Index g = 0; g < G; ++g) {
        const auto& b = bdry[static_cast<std::size_t>(g)];
        h_a(n_v + g) = 0.5 * (h_a(b.v0) + h_a(b.v1));
    }

    constexpr double gamma = 0.8;           // paper's shell value (W-D2)
    Eigen::VectorXd beta(n_ext);
    for (Eigen::Index a = 0; a < n_ext; ++a) {
        beta(a) = gamma / (h_a(a) * h_a(a));
    }
    const double r_cut =
        std::sqrt(-std::log(1e-10) / gamma) * h_a.maxCoeff();

    // --- conforming scalar Kirchhoff assembly ------------------------
    const double D_bend = sm.k_B;
    const double rho_h  = mat.density * h;
    const double nu     = mat.poisson_ratio;
    Eigen::Matrix3d C_bend;
    C_bend << 1.0, nu,  0.0,
              nu,  1.0, 0.0,
              0.0, 0.0, 2.0 * (1.0 - nu);

    // Assemble at quadrature-CELL subdivision levels: level 0 = the
    // mesh triangles (12-pt rule = what the assembler does), level L =
    // each triangle 4^L-way midpoint-split, 12-pt rule per sub-cell.
    // Same basis, same nodes — PURE integration refinement. The rule
    // ORDER was measured insensitive (7 vs 12 pt); this probes
    // integration RESOLUTION against the sub-h oscillation of the
    // sublattice-zigzag integrand.
    const auto qpts = chladni::shell::quadrature_points(
        chladni::shell::QuadratureRule::TwelvePointDunavant);
    const auto assemble_level = [&](int level,
                                    Eigen::MatrixXd& Kz,
                                    Eigen::MatrixXd& Mz) {
        Kz.setZero(n_ext, n_ext);
        Mz.setZero(n_ext, n_ext);
        // Recursive 4-way midpoint split down to `level`.
        const std::function<void(const Eigen::Vector2d&,
                                 const Eigen::Vector2d&,
                                 const Eigen::Vector2d&, int)>
            integrate_tri = [&](const Eigen::Vector2d& A,
                                const Eigen::Vector2d& B,
                                const Eigen::Vector2d& C, int lvl) {
            if (lvl > 0) {
                const Eigen::Vector2d AB = 0.5 * (A + B);
                const Eigen::Vector2d BC = 0.5 * (B + C);
                const Eigen::Vector2d CA = 0.5 * (C + A);
                integrate_tri(A, AB, CA, lvl - 1);
                integrate_tri(AB, B, BC, lvl - 1);
                integrate_tri(CA, BC, C, lvl - 1);
                integrate_tri(AB, BC, CA, lvl - 1);
                return;
            }
            const Eigen::Vector2d e01 = B - A, e02 = C - A;
            const double two_area =
                std::abs(e01.x() * e02.y() - e01.y() * e02.x());
            for (const auto& q : qpts) {
                const Eigen::Vector2d xg =
                    (1.0 - q.v - q.w) * A + q.v * B + q.w * C;
                const double w_dA = q.weight * two_area;
                const auto gh =
                    chladni::shell::lme::evaluate_basis_grad_and_hess(
                        nodes2d, beta, xg, r_cut, 1e-10, 60);
                const std::size_t n_act = gh.indices.size();
                std::vector<Eigen::Vector3d> Hv(n_act);
                for (std::size_t k = 0; k < n_act; ++k) {
                    const auto& H = gh.hessians[k];
                    Hv[k] =
                        Eigen::Vector3d(H(0, 0), H(1, 1), H(0, 1));
                }
                for (std::size_t i = 0; i < n_act; ++i) {
                    const int a = gh.indices[i];
                    for (std::size_t j = 0; j < n_act; ++j) {
                        const int b = gh.indices[j];
                        Kz(a, b) += w_dA * D_bend *
                            Hv[i].dot(C_bend * Hv[j]);
                        Mz(a, b) += w_dA * rho_h *
                            gh.values[i] * gh.values[j];
                    }
                }
            }
        };
        for (Eigen::Index t = 0; t < mesh.F.rows(); ++t) {
            integrate_tri(nodes2d.row(mesh.F(t, 0)),
                          nodes2d.row(mesh.F(t, 1)),
                          nodes2d.row(mesh.F(t, 2)), level);
        }
    };

    Eigen::MatrixXd Kz, Mz;
    for (int level = 0; level <= 2; ++level) {
        assemble_level(level, Kz, Mz);
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges_l(
            0.5 * (Kz + Kz.transpose()), 0.5 * (Mz + Mz.transpose()),
            Eigen::EigenvaluesOnly);
        std::cout << "[quadsplit_conforming] level " << level
                  << " (" << (qpts.size() << (2 * level))
                  << " pts/tri) lowest 10 (Hz):";
        for (Eigen::Index i = 0; i < 10; ++i) {
            const double lam = std::max(ges_l.eigenvalues()(i), 0.0);
            std::cout << " " << std::sqrt(lam) / two_pi;
        }
        std::cout << "\n";
    }
    // Keep the level-0 operators for the cross-energy check below
    // (level 0 = the assembler's own integration resolution).
    assemble_level(0, Kz, Mz);

    // --- conforming spectrum (level 0) --------------------------------
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        0.5 * (Kz + Kz.transpose()), 0.5 * (Mz + Mz.transpose()),
        Eigen::ComputeEigenvectors);

    // --- curved pencil's spurious mode under the conforming energy ---
    chladni::shell::LMEAssembler::Params p;
    p.use_curved_shell = true;
    p.use_ghost_nodes  = true;
    chladni::shell::LMEAssembler asm_(p);
    const Eigen::SparseMatrix<double> Kc =
        asm_.assemble_K(mesh.V, mesh.F, sm);
    const Eigen::SparseMatrix<double> Mc =
        asm_.assemble_M(mesh.V, mesh.F, mat.density * h);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges_c(
        Eigen::MatrixXd(Kc), Eigen::MatrixXd(Mc),
        Eigen::ComputeEigenvectors);
    const Eigen::VectorXd vc = ges_c.eigenvectors().col(6);
    const double f_curved =
        std::sqrt(std::max(ges_c.eigenvalues()(6), 0.0)) / two_pi;
    Eigen::VectorXd vz(n_ext);
    for (Eigen::Index a = 0; a < n_ext; ++a) vz(a) = vc(3 * a + 2);
    const double e_conf = vz.dot(Kz * vz);
    const double m_conf = vz.dot(Mz * vz);
    const double f_conf = std::sqrt(e_conf / m_conf) / two_pi;
    std::cout << "[quadsplit_conforming] curved mode[6] f=" << f_curved
              << " Hz; SAME z-coefficients under conforming pencil: f="
              << f_conf << " Hz  (energy ratio "
              << (e_conf / m_conf)
                     / std::pow(two_pi * f_curved, 2)
              << ")\n";
    SUCCEED();
}

TEST_CASE("LME curved+ghost+SME on 32x8 polar disk vs Leissa — Phase B.2 bar",
          "[.slow][lme][circular][curved][ghost][sme][validation]")
{
    // 2nd-order SME (Rosolen 2013) regression bar. HISTORY: SME was
    // originally motivated by a "~7.9 % 1st-order LME ceiling" on this
    // fixture (pinned 2026-05-22, commit 8664494); that ceiling was
    // FALSIFIED 2026-06-03 — it was the uniform-β closed-form
    // derivative bug (fixed per Millán 2011 App A, `edaf0e1`), and
    // 1st-order LME now reads +0.75 % here. SME measures an
    // equivalent +0.73 % (2026-06-04); the bar still guards the SME
    // wire-up (chart-boundary classification, slack matrices, wide
    // r_cut dispatch) rather than an accuracy win on this fixture —
    // SME's edge is membrane-dominated curved shells. Gate at 2 %
    // matches the 1st-order gate above.
    //
    // [.slow] because SME's 5-dim Newton + IFT + FD-on-grad Hessian
    // per Gauss point makes curved K assembly ~3-5x slower than the
    // 1st-order LME path. 32x8 disk is the smallest fixture where Leissa's
    // analytic prediction is tight enough for a meaningful
    // accuracy gate.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto mesh = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;

    chladni::shell::LMEAssembler::Params p;
    p.use_curved_shell      = true;
    p.use_ghost_nodes       = true;
    p.use_second_order_sme  = true;
    // No r_cut override: this exercises the SAME default path the GUI
    // uses. Since 2026-06-04 (inventory C7) the SME truncation is
    // VALUE-based from the paper's own decay rate γ_eff = 2/α —
    // r = h·√(ln(1/TOL_LME)·α/2) ≈ 4.80 h at the shipped α=2 — which
    // lands on the retired r_cut_mult_sme constant's measured
    // [2.5, 4.0] plateau (that field is no longer read). The historic
    // sweep (2026-05-27): shallow ~0.1 % plateau over [2.5, 4.0],
    // 1.2 % at 1.6, DIVERGED at 1.4 — what crashed the GUI before SME
    // stopped sharing LME's 1.4 default. SME's accuracy lever is its
    // slack matrix, not truncation.
    chladni::shell::LMEAssembler asm_(p);

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
    const double f_n2 = 0.5 *
        (modes.omegas(0) + modes.omegas(1)) / two_pi;
    const double rel_err =
        std::abs(f_n2 - f_n2_ref) / f_n2_ref;
    std::cout << "[SME 32x8 polar disk, Leissa n=2 = " << f_n2_ref
              << " Hz, SME n=2 = " << f_n2
              << " Hz, rel_err = " << rel_err << "]\n";

    REQUIRE(rel_err < 0.02);

    // Full 7-mode pin through the (1,1) doublet (added 2026-06-06;
    // same cluster indexing as the 1st-order shipped-defaults gate —
    // see its comment). Measured at the shipped SME defaults
    // ([disk_table] 2026-06-06): (0,1) +0.66%, (3,0) +1.03%,
    // (1,1) +1.15%. Same 2% bar.
    struct Cluster { const char* label; int i0; int i1; };
    for (const Cluster c : {Cluster{"(0,1)", 2, 2},
                            Cluster{"(3,0)", 3, 4},
                            Cluster{"(1,1)", 5, 6}}) {
        const double f_ref = 0.5
            * (analytic[static_cast<std::size_t>(c.i0)]
               + analytic[static_cast<std::size_t>(c.i1)]) / two_pi;
        const double f = 0.5
            * (modes.omegas(c.i0) + modes.omegas(c.i1)) / two_pi;
        const double err = std::abs(f - f_ref) / f_ref;
        std::cout << "[SME 32x8 polar disk, Leissa " << c.label
                  << " = " << f_ref << " Hz, SME = " << f
                  << " Hz, rel_err = " << err << "]\n";
        REQUIRE(err < 0.02);
    }
}

TEST_CASE("LME curved+ghost r_cut_mult sweep on 32x8 polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][rcut_sweep]")
{
    // Diagnostic — 1st-order LME free-edge accuracy on the polar
    // disk, swept over r_cut_mult across three resolutions.
    //
    // STALE-FINDING REFRESH (2026-06-03): the original sweet-spot
    // claim ("r_cut_mult ≈ 1.32 drops 32x8 n=2 from 7.9 % to ~0.2 %",
    // from the γ=1.6 era) NO LONGER HOLDS, and not because of a
    // regression: the curved LME path now sizes its chart from the
    // paper's value-based geometric support (Millán Eq. 2, derived
    // from chart_tol_lme / tol_lme), which SUPERSEDES the legacy
    // r_cut_mult_curved truncation. So r_cut_mult is now a no-op on
    // this path and the sweep is FLAT — every rcm in [1.2, 4.0]
    // gives the same value. That flatness is the thing this
    // diagnostic now guards. The genuine accuracy lever is
    // refinement: re-measured 2026-06-04 under the per-node-β
    // derivatives (`edaf0e1`), n=2 rel_err runs 16x3 ≈ 3.8 %,
    // 32x8 ≈ 0.75 %, 64x16 ≈ 0.12 % — clean O(h²)-ish convergence
    // to the analytic. (The 2026-06-03 refresh quoted 17 / 7.35 /
    // 2.0 % — measured hours before the derivative fix landed.)
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;

    std::cout.setf(std::ios::unitbuf);
    struct Level { int n_az; int n_rad; };
    for (Level L : {Level{16, 3}, Level{32, 8}, Level{64, 16}}) {
        const auto mesh = chladni::mesh::generate_circular_disk(
            R, L.n_az, L.n_rad);
        const auto analytic =
            chladni::analytical::free_edge_circular_plate_angular_frequencies(
                {.radius = R, .thickness = h}, mat, n_modes);
        const double f_n2_ref = analytic[0] / two_pi;
        std::cout << "[r_cut_mult-sweep " << L.n_az << "x" << L.n_rad
                  << " V=" << mesh.V.rows()
                  << ", Leissa n=2 = " << f_n2_ref << " Hz]\n";
        for (double rcm :
             {1.20, 1.25, 1.30, 1.32, 1.34, 1.36, 1.40, 1.5, 2.0, 4.0}) {
            chladni::shell::LMEAssembler::Params p;
            p.use_curved_shell = true;
            p.use_ghost_nodes  = true;
            p.r_cut_mult       = rcm;
            try {
                chladni::shell::LMEAssembler asm_(p);
                const auto modes = chladni::shell::compute_shell_modes(
                    mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
                const double f_n2 = 0.5 *
                    (modes.omegas(0) + modes.omegas(1)) / two_pi;
                std::cout << "  r_cut_mult=" << rcm
                          << "  n=2: " << f_n2
                          << " (rel_err "
                          << ((f_n2 - f_n2_ref) / f_n2_ref) << ")"
                          << '\n';
            } catch (const std::exception& e) {
                std::cout << "  r_cut_mult=" << rcm << "  FAILED — "
                          << e.what() << '\n';
            }
        }
    }
}

TEST_CASE("LME curved+ghost r_cut_mult sweep on closed icosphere k=2 vs Wilkinson",
          "[.diag][lme][icosphere][curved][rcut_sweep_sphere]")
{
    // Cross-check: does the polar-disk's r_cut_mult=1.32 sweet spot
    // generalise to closed shells? LME default r_cut_mult=4 gives
    // ~2.25 % on icosphere k=2 n=2 spheroidal pentet vs Wilkinson
    // (per [[chladni-stiffness-alternatives]] §h). If the sweep
    // moves this in the right direction we have a robust new
    // default; if it makes it worse, the new optimum is fixture-
    // dependent and we need adaptive selection.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 6;
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);
    const auto mesh = chladni::mesh::generate_icosphere(R, 2);
    const auto wilkinson = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel, /*n_modes=*/1);
    const double w_n2_ref = wilkinson.front();
    std::cout.setf(std::ios::unitbuf);
    std::cout << "[r_cut_mult-sweep icosphere k=2 V=" << mesh.V.rows()
              << ", Wilkinson n=2 = " << w_n2_ref << " rad/s]\n";
    for (double rcm :
         {1.20, 1.32, 1.5, 2.0, 3.0, 4.0, 6.0}) {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = false;  // closed mesh, no ghosts
        p.r_cut_mult       = rcm;
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, steel, sm, h, n_modes, asm_);
            double sum = 0.0;
            double wmin = modes.omegas(0), wmax = modes.omegas(0);
            for (Eigen::Index i = 0; i < 5; ++i) {
                const double w = modes.omegas(i);
                sum += w;
                wmin = std::min(wmin, w);
                wmax = std::max(wmax, w);
            }
            const double pentet_mean = sum / 5.0;
            const double cluster_spread = (wmax - wmin) / pentet_mean;
            const double rel_err =
                std::abs(pentet_mean - w_n2_ref) / w_n2_ref;
            std::cout << "  r_cut_mult=" << rcm
                      << "  pentet_mean=" << pentet_mean
                      << "  rel_err=" << rel_err
                      << "  cluster_spread=" << cluster_spread
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << "  r_cut_mult=" << rcm << "  FAILED — "
                      << e.what() << '\n';
        }
    }
}

TEST_CASE("LME curved+ghost k_ring sweep on 32x8 polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][kring_sweep]")
{
    // Last untried LME knob. Default k_ring_depth=3 produces ~37-node
    // charts on this fixture. kRing=2 ~10-node, kRing=4 known broken
    // on 72x8 (non-PSD K) but might work at 32x8 with smaller centre
    // valence. Pinned at r_cut_mult_curved=1.4 (new curved-path
    // default) so the result reads against the current accuracy
    // baseline.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto mesh = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;
    std::cout.setf(std::ios::unitbuf);
    std::cout << "[k_ring-sweep 32x8 polar disk, Leissa n=2 = "
              << f_n2_ref << " Hz]\n";
    for (int kring : {2, 3, 4, 5}) {
        chladni::shell::LMEAssembler::Params p;
        p.k_ring_depth     = kring;
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
            const double f_n2 = 0.5 *
                (modes.omegas(0) + modes.omegas(1)) / two_pi;
            std::cout << "  k_ring_depth=" << kring
                      << "  n=2: " << f_n2
                      << " (rel_err " << ((f_n2 - f_n2_ref) / f_n2_ref) << ")"
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << "  k_ring_depth=" << kring << "  FAILED — "
                      << e.what() << '\n';
        }
    }
}

TEST_CASE("LME curved+ghost gamma_pu sweep on 32x8 polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][gamma_pu_sweep]")
{
    // Diagnostic — gamma_pu is the Shepard PoU aspect ratio
    // (Millán 2011 Table I band [3, 6], default 4). Tighter
    // PoU = sharper patch blending = less smoothing across charts;
    // looser = more blending. Independent knob from γ_LME.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto mesh = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;
    std::cout << "[gamma_pu-sweep 32x8 polar disk, Leissa n=2 = "
              << f_n2_ref << " Hz]\n";
    std::cout.setf(std::ios::unitbuf);
    for (double gp : {2.0, 3.0, 4.0, 5.0, 6.0, 8.0}) {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = true;
        p.gamma_pu         = gp;
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
            const double f_n2 = 0.5 *
                (modes.omegas(0) + modes.omegas(1)) / two_pi;
            std::cout << "  gamma_pu=" << gp
                      << "  n=2: " << f_n2
                      << " (rel_err " << ((f_n2 - f_n2_ref) / f_n2_ref) << ")"
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << "  gamma_pu=" << gp << "  FAILED — "
                      << e.what() << '\n';
        }
    }
}

TEST_CASE("LME curved+ghost γ-sweep on 32x8 polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][gamma_sweep]")
{
    // Diagnostic. Rosolen 2013 abstract claims 1st-order LME with
    // LARGE supports (small γ) is competitive with 2nd-order SME on
    // biharmonic problems — and post-derivative-fix that is exactly
    // what we see. Re-measured 2026-06-04 (per-node-β derivatives,
    // `edaf0e1`; shipped default γ=0.8 per the W-D2 faithfulness fix):
    //   γ=0.8   n=2 +0.75 %  (the default-path validation gate above)
    //   γ=1.6   n=2 +4.3 %, n=0,s=1 +5.2 %, n=3 +2.6 %
    //   γ=2.5   n=2 +34 %,  n=0,s=1 +26 %,  n=3 +29 %
    // i.e. WIDER support is monotonically better; sharpening γ
    // degrades. (Pre-fix this sweep read γ=1.6 → 7.9 % / γ=2.5 →
    // 29.6 % and was used to argue "larger support does NOT close the
    // gap" — that conclusion was an artifact of the uniform-β
    // derivative bug. The γ=4.0 ABORT — every Shepard patch's
    // in-chart Newton diverging — is a pre-fix observation, not
    // re-checked since.)
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto mesh = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;
    const double f_n0_ref = analytic[2] / two_pi;
    const double f_n3_ref = analytic[3] / two_pi;
    std::cout << "[γ-sweep 32x8 polar disk, Leissa n=2 = " << f_n2_ref
              << " Hz, n=0,s=1 = " << f_n0_ref
              << " Hz, n=3 = " << f_n3_ref << " Hz]\n";
    std::cout.setf(std::ios::unitbuf);
    for (double gamma : {1.6, 2.5}) {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = true;
        p.gamma            = gamma;
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
            const double f_n2 = 0.5 *
                (modes.omegas(0) + modes.omegas(1)) / two_pi;
            const double f_n0 = modes.omegas(2) / two_pi;
            const double f_n3 = 0.5 *
                (modes.omegas(3) + modes.omegas(4)) / two_pi;
            std::cout << "  γ=" << gamma
                      << "  n=2: " << f_n2
                      << " (rel_err " << ((f_n2 - f_n2_ref) / f_n2_ref) << ")"
                      << "  n=0,s=1: " << f_n0
                      << " (rel_err " << ((f_n0 - f_n0_ref) / f_n0_ref) << ")"
                      << "  n=3: " << f_n3
                      << " (rel_err " << ((f_n3 - f_n3_ref) / f_n3_ref) << ")"
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << "  γ=" << gamma << "  FAILED — " << e.what() << '\n';
        }
    }
}

TEST_CASE("LME curved+ghost mesh-refinement sweep on free-edge polar disk vs Leissa",
          "[.diag][lme][circular][curved][ghost][refinement_sweep]")
{
    // Diagnostic — not a regression gate. Asks whether mesh refinement
    // converges the free-edge fixture to the analytic. Re-measured
    // 2026-06-04 (per-node-β derivatives, `edaf0e1`): n=2 rel_err
    // 16x3 +3.8 %, 32x8 +0.75 %, 64x16 +0.12 % — clean monotonic
    // O(h²)-ish convergence; refinement IS the lever. (The pre-fix
    // sweep read a non-monotonic 14.6 / 7.9 / 11.6 % and was part of
    // the evidence for the falsified "~8 % 1st-order floor".)
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, n_modes);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref = analytic[0] / two_pi;
    std::cout << "[refinement sweep, Leissa n=2 = " << f_n2_ref << " Hz]\n";
    std::cout.setf(std::ios::unitbuf);
    struct Level { int n_az; int n_rad; };
    for (Level L : {Level{16, 3}, Level{32, 8}, Level{64, 16}}) {
        const auto mesh = chladni::mesh::generate_circular_disk(R, L.n_az, L.n_rad);
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = true;
        p.use_ghost_nodes  = true;
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, n_modes, asm_);
            const double f_n2 = 0.5 *
                (modes.omegas(0) + modes.omegas(1)) / two_pi;
            std::cout << "  " << L.n_az << "x" << L.n_rad
                      << " V=" << mesh.V.rows()
                      << "  n=2: " << f_n2
                      << " (rel_err " << ((f_n2 - f_n2_ref) / f_n2_ref) << ")"
                      << '\n';
        } catch (const std::exception& e) {
            std::cout << "  " << L.n_az << "x" << L.n_rad
                      << "  FAILED — " << e.what() << '\n';
        }
    }
}

TEST_CASE("LME curved K + M with ghost nodes — extended-size assembly",
          "[lme][circular][curved][ghost][validation]")
{
    // Ghost-nodes-on path (Millán 2011 §4.1.2). The extended LME node
    // set has N + G nodes and K, M are sized 3*(N+G). With ghosts in
    // place the boundary-adjacent Gauss points all sit inside the
    // extended convex hull of LME nodes, so the in-chart Newton stays
    // convergent on the rim-adjacent patches.
    //
    // 16x3 = 49 V, 16 boundary edges. Small enough to bisect bugs
    // quickly; large enough to exercise multiple boundary-near anchors
    // with ghost neighbours.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr int    n_az  = 16;
    constexpr int    n_rad = 3;

    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    const auto mesh = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();
    constexpr Eigen::Index G = n_az;  // 16 boundary edges

    chladni::shell::LMEAssembler::Params p;
    p.use_curved_shell = true;
    p.use_ghost_nodes  = true;
    chladni::shell::LMEAssembler asm_(p);

    auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
    auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    REQUIRE(K.rows() == 3 * (n_v + G));
    REQUIRE(K.cols() == 3 * (n_v + G));
    REQUIRE(M.rows() == 3 * (n_v + G));
    REQUIRE(M.cols() == 3 * (n_v + G));

    const Eigen::MatrixXd Kd = Eigen::MatrixXd(K);
    const Eigen::MatrixXd Md = Eigen::MatrixXd(M);
    REQUIRE(Kd.norm() > 0.0);
    REQUIRE(Md.norm() > 0.0);
    REQUIRE((Kd - Kd.transpose()).norm() / Kd.norm() < 1.0e-10);
    REQUIRE((Md - Md.transpose()).norm() / Md.norm() < 1.0e-10);
}

TEST_CASE("LME ghost-nodes assemble on the 32x4 polar disk (kRing-reach gap)",
          "[lme][circular][curved][ghost][validation]")
{
    // Companion to the 16x3 ghost-on test above. The 32x4 fixture
    // has 4 radial rings; ring-1 anchor's kRing=3 BFS reaches the
    // rim at depth 3, so ghosts at depth 4 fall outside its patch
    // — without the tol_PU=1e-3 cull (guarded by use_ghost_nodes)
    // ring-1 patches would still be Shepard-active at near-rim
    // Gauss points with charts that don't see the ghost extension,
    // and the in-chart Newton would diverge.
    //
    // Pins that the combined ghost + tightened-tol_PU pipeline
    // produces a well-formed K and M of size 3*(N+G) on this
    // historically-pathological fixture.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr int    n_az  = 32;
    constexpr int    n_rad = 4;

    const auto mat   = steel_033();
    const auto sm    = chladni::shell::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    const auto mesh  = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();
    constexpr Eigen::Index G = n_az;  // 32 rim edges

    chladni::shell::LMEAssembler::Params p;
    p.use_curved_shell = true;
    p.use_ghost_nodes  = true;
    chladni::shell::LMEAssembler asm_(p);

    auto K = asm_.assemble_K(mesh.V, mesh.F, sm);
    auto M = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    REQUIRE(K.rows() == 3 * (n_v + G));
    REQUIRE(M.rows() == 3 * (n_v + G));
}

TEST_CASE("LME ghost-nodes eigensolve end-to-end on a 16x3 polar disk",
          "[lme][circular][curved][ghost][validation]")
{
    // Full pipeline: assemble (3*(N+G)-sized K, M) → eigensolve →
    // evaluate_modes_at_vertices slices trailing 3*G rows of each
    // eigenvector. Verifies (a) mode shapes return at 3*N rows, (b)
    // the eigensolve still converges on the ghost-extended system, and
    // (c) ghost-ON is a clear IMPROVEMENT over ghost-OFF toward the
    // analytic free-edge plate (ghosts cure the §4.1.2 boundary
    // flattening; the two are NOT expected to agree).
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr int    n_az  = 16;
    constexpr int    n_rad = 3;
    constexpr std::size_t n_modes = 6;

    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h_thk);
    const auto mesh = chladni::mesh::generate_circular_disk(R, n_az, n_rad);
    const Eigen::Index n_v = mesh.V.rows();

    // Ghost-on
    chladni::shell::LMEAssembler::Params p_ghost;
    p_ghost.use_curved_shell = true;
    p_ghost.use_ghost_nodes  = true;
    chladni::shell::LMEAssembler asm_ghost(p_ghost);

    const auto modes_ghost = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, mat, sm, h_thk, n_modes, asm_ghost);

    REQUIRE(modes_ghost.shapes.rows() == 3 * n_v);
    REQUIRE(modes_ghost.shapes.cols() == static_cast<Eigen::Index>(n_modes));
    REQUIRE(modes_ghost.omegas.size() ==
            static_cast<Eigen::Index>(n_modes));
    for (Eigen::Index i = 0; i < modes_ghost.omegas.size(); ++i) {
        REQUIRE(modes_ghost.omegas(i) > 0.0);  // physical mode, not rigid
    }

    // Ghost-off baseline for cross-check.
    chladni::shell::LMEAssembler::Params p_off;
    p_off.use_curved_shell = true;
    p_off.use_ghost_nodes  = false;
    chladni::shell::LMEAssembler asm_off(p_off);

    const auto modes_off = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, mat, sm, h_thk, n_modes, asm_off);

    REQUIRE(modes_off.shapes.rows() == 3 * n_v);

    std::cout << "[lme ghost eigensolve] freqs (Hz):\n";
    for (Eigen::Index i = 0; i < modes_ghost.omegas.size(); ++i) {
        const double f_ghost = modes_ghost.omegas(i) / (2.0 * M_PI);
        const double f_off   = modes_off.omegas(i)   / (2.0 * M_PI);
        std::cout << "  mode " << i << ": off=" << f_off
                  << " ghost=" << f_ghost
                  << " (delta " << (f_ghost - f_off) / f_off * 100.0
                  << "%)\n";
    }

    // Ghost nodes exist to CURE the §4.1.2 free-edge basis flattening (the
    // LME approximants go asymptotically flat at a free boundary → spurious
    // membrane stiffness → the spectrum locks high). So ghost-ON and ghost-OFF
    // are NOT expected to agree — ghost-ON must be a clear IMPROVEMENT toward
    // the analytic free-edge plate, and ghost-OFF should lock hard. (The old
    // "<30% consistency" bar wrongly assumed the two would be close; it only
    // held by luck at the over-sharp γ_LME=1.6 + fixed k-ring, where ghost-OFF
    // locked less.) Validate the improvement against Leissa.
    const auto analytic =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h_thk}, mat, 2);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_n2_ref   = analytic[0] / two_pi;
    const double f_n2_ghost =
        0.5 * (modes_ghost.omegas(0) + modes_ghost.omegas(1)) / two_pi;
    const double f_n2_off =
        0.5 * (modes_off.omegas(0) + modes_off.omegas(1)) / two_pi;
    const double rel_ghost = std::abs(f_n2_ghost - f_n2_ref) / f_n2_ref;
    const double rel_off   = std::abs(f_n2_off   - f_n2_ref) / f_n2_ref;
    INFO("n=2  Leissa = " << f_n2_ref
         << " Hz | ghost-ON = " << f_n2_ghost << " (rel " << rel_ghost
         << ") | ghost-OFF = " << f_n2_off << " (rel " << rel_off << ")");

    // Ghost-ON lands at +3.8 % on this very coarse 16x3 disk (49 V; measured
    // 2026-06-04 under the per-node-β derivatives — the pre-fix value here
    // was ~18 %), converging O(h²)-ish under refinement (32x8 → +0.75 %).
    REQUIRE(rel_ghost < 0.08);
    // Ghost-OFF locks hard at the free edge (basis flattening), so ghost-ON is
    // a large, unambiguous improvement — at least 2x closer to the analytic.
    REQUIRE(rel_ghost < 0.5 * rel_off);
}
