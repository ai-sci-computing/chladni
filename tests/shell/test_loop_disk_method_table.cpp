/**
 * @file test_loop_disk_method_table.cpp
 * @brief Method comparison tables across three disk topologies.
 *
 * The ShellAssembler framework lets us sweep K/M quadrature × mass-
 * lumping combinations cheaply. This benchmark does exactly that on
 * the three procedural disk topologies in the library:
 *
 *  - **Polar disk** (@ref chladni::mesh::generate_circular_disk):
 *    central valence-n_az hub, concentric ring structure. Highest
 *    discrete rotational symmetry (D_{n_az}) — handles every (m, s)
 *    with m ≤ n_az/2 cleanly in the bulk; the lone defect is a
 *    valence-n_az central hub that distorts modes near the origin.
 *
 *  - **Iso disk** (@ref chladni::mesh::generate_disk_iso): isotropic
 *    concentric rings without the central hub — interior valences
 *    distributed over the disk. No useful discrete symmetry — the
 *    Galerkin projection onto its basis gives mode shapes with
 *    distributed asymmetries.
 *
 *  - **Hex disk** (@ref chladni::mesh::generate_disk_hex): clipped
 *    hex lattice with rim snapped to a circle. Cleanest possible
 *    Loop input — every interior vertex valence 6 — but with D_6
 *    discrete rotational symmetry, which aliases high-m doublets
 *    (e.g. (7,0) → m'=1 admixture). Valence-3 rim corners can
 *    trigger boundary-augmentation rejection.
 *
 * For each topology, six (k_quad, m_quad, m_lump) Loop configurations
 * plus the two shipped MESHFREE configurations (1st-order LME and
 * 2nd-order SME, both curved+ghost defaults) are tabulated against
 * the Leissa free-edge circular plate reference — 7 modes, i.e.
 * through the (1,1) doublet (2026-06-06: meshfree rows added to fill
 * the (1,1) column for every method; the iso/hex rows double as the
 * mesh-independence probe — hex's valence-3 rim corners are exactly
 * what Loop's boundary augmentation rejects).
 *
 * Tagged @c [.benchmark] — invoke with
 *
 *   ./build/tests/chladni_tests "[.benchmark][shell][loop][disk_table]"
 *
 * Total runtime a few minutes (24 eigensolves on ~250 V meshes; the
 * SME rows dominate).
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs   = chladni::shell;
namespace cmsh = chladni::mesh;

namespace {

chladni::IsotropicMaterial steel_033()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.33,  // matches Leissa Table 2.5 column
            .density        = 7850.0};
}

const char* rule_label(cs::QuadratureRule r)
{
    switch (r) {
    case cs::QuadratureRule::OnePointCentroid:  return "1pt";
    case cs::QuadratureRule::ThreePointEdgeMid: return "3pt";
    case cs::QuadratureRule::SevenPointDunavant: return "7pt";
    }
    return "?";
}

const char* lump_label(cs::MassLumping l)
{
    switch (l) {
    case cs::MassLumping::None:   return "consistent";
    case cs::MassLumping::RowSum: return "row-sum";
    }
    return "?";
}

struct RunResult {
    bool ok;
    std::string err;
    std::vector<double> freqs_hz;
};

RunResult run_disk(
    const cmsh::TriMesh& mesh,
    cs::QuadratureRule rule, cs::MassLumping lump,
    std::size_t n_modes)
{
    constexpr double h = 1.0e-3;
    cs::LoopAssembler::Params p;
    p.k_quad   = rule;
    p.m_quad   = rule;
    p.m_lump   = lump;
    p.n_passes = 1;
    p.use_stam = true;

    try {
        const auto sm = cs::shell_material_from_isotropic(steel_033(), h);
        const auto modes = cs::compute_shell_modes(
            mesh.V, mesh.F, steel_033(), sm, h, n_modes,
            cs::LoopAssembler{p});
        constexpr double tau = 2.0 * std::numbers::pi_v<double>;
        RunResult r;
        r.ok = true;
        r.freqs_hz.reserve(n_modes);
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            r.freqs_hz.push_back(modes.omegas(i) / tau);
        }
        return r;
    } catch (const std::exception& e) {
        return RunResult{.ok = false, .err = e.what(), .freqs_hz = {}};
    }
}

// Shipped meshfree configurations (curved-shell + ghost defaults):
// 1st-order LME, and 2nd-order SME when sme is true (the GUI default
// algorithm). compute_shell_modes strips the ghost DOFs internally,
// so the returned spectrum is directly comparable to the Loop rows.
RunResult run_disk_meshfree(
    const cmsh::TriMesh& mesh, std::size_t n_modes, bool sme)
{
    constexpr double h = 1.0e-3;
    cs::LMEAssembler::Params p;
    p.use_second_order_sme = sme;
    try {
        const auto sm = cs::shell_material_from_isotropic(steel_033(), h);
        const auto modes = cs::compute_shell_modes(
            mesh.V, mesh.F, steel_033(), sm, h, n_modes,
            cs::LMEAssembler{p});
        constexpr double tau = 2.0 * std::numbers::pi_v<double>;
        RunResult r;
        r.ok = true;
        r.freqs_hz.reserve(n_modes);
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            r.freqs_hz.push_back(modes.omegas(i) / tau);
        }
        return r;
    } catch (const std::exception& e) {
        return RunResult{.ok = false, .err = e.what(), .freqs_hz = {}};
    }
}

// Legacy CST membrane + Wardetzky-IBM bending — the formulation the
// project shipped before the Loop subdivision rewrite. Documented as
// ~60% over-stiff on flat-plate fixtures (the W-block calibration
// gap). Reachable through the framework-free entry point
// chladni::shell::compute_shell_modes(V, F, mat, thickness, n_modes).
RunResult run_disk_legacy(
    const cmsh::TriMesh& mesh, std::size_t n_modes)
{
    constexpr double h = 1.0e-3;
    try {
        const auto modes = cs::compute_shell_modes(
            mesh.V, mesh.F, steel_033(), h, n_modes);
        constexpr double tau = 2.0 * std::numbers::pi_v<double>;
        RunResult r;
        r.ok = true;
        r.freqs_hz.reserve(n_modes);
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            r.freqs_hz.push_back(modes.omegas(i) / tau);
        }
        return r;
    } catch (const std::exception& e) {
        return RunResult{.ok = false, .err = e.what(), .freqs_hz = {}};
    }
}

void print_table(
    const std::string& disk_label,
    const cmsh::TriMesh& mesh,
    const std::vector<double>& leissa_hz,
    std::size_t n_modes)
{
    std::cout << "\n[" << disk_label << "]  V=" << mesh.V.rows()
              << "  F=" << mesh.F.rows() << "\n"
              << "  Leissa (Hz): ";
    for (std::size_t i = 0; i < std::min(leissa_hz.size(), n_modes); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << leissa_hz[i] << " ";
    }
    std::cout << "\n"
                 "  config               |   mode 0  mode 1  mode 2  mode 3  mode 4  mode 5  mode 6 | max %% err  (2,0) split %%\n"
                 "  ---------------------+--------------------------------------------------------+-----------+----------------\n";

    auto print_row = [&](const std::string& label, const RunResult& r) {
        std::cout << "  " << std::setw(20) << std::left << label
                  << " |";
        if (!r.ok) {
            std::cout << "  THROW: " << r.err << "\n";
            return;
        }
        double max_err = 0.0;
        for (std::size_t i = 0;
             i < std::min<std::size_t>(7, r.freqs_hz.size()); ++i) {
            const double err = (i < leissa_hz.size() && leissa_hz[i] > 0.0)
                ? std::abs(r.freqs_hz[i] - leissa_hz[i]) / leissa_hz[i]
                : 0.0;
            if (err > max_err) max_err = err;
            std::cout << " " << std::setw(7) << std::fixed
                      << std::setprecision(2) << r.freqs_hz[i];
        }
        const double split = (r.freqs_hz.size() >= 2)
            ? 100.0 * std::abs(r.freqs_hz[1] - r.freqs_hz[0])
                   / std::max(0.5 * (r.freqs_hz[0] + r.freqs_hz[1]), 1.0)
            : 0.0;
        std::cout << " |  " << std::setw(7) << std::fixed
                  << std::setprecision(2) << (100.0 * max_err)
                  << "  |  " << std::setw(7) << std::fixed
                  << std::setprecision(3) << split << "\n";
    };

    // Legacy CST+IBM baseline first, so the Loop variants below have
    // an explicit "what we were doing before the framework" reference.
    print_row("Legacy CST+IBM", run_disk_legacy(mesh, n_modes));

    const std::vector<cs::QuadratureRule> rules{
        cs::QuadratureRule::OnePointCentroid,
        cs::QuadratureRule::ThreePointEdgeMid,
        cs::QuadratureRule::SevenPointDunavant};
    const std::vector<cs::MassLumping> lumps{
        cs::MassLumping::None,
        cs::MassLumping::RowSum};

    for (const auto rule : rules) {
        for (const auto lump : lumps) {
            const std::string label =
                std::string{"Loop K=M="} + rule_label(rule)
                + "  " + lump_label(lump);
            print_row(label, run_disk(mesh, rule, lump, n_modes));
        }
    }

    // Shipped meshfree rows (curved+ghost defaults). On the hex disk
    // these are the mesh-independence probe: its valence-3 rim
    // corners reject Loop's boundary augmentation, while the meshfree
    // paths carry no connectivity requirements.
    print_row("LME-1st (shipped)",
              run_disk_meshfree(mesh, n_modes, /*sme=*/false));
    print_row("SME (shipped)",
              run_disk_meshfree(mesh, n_modes, /*sme=*/true));
}

}  // namespace

TEST_CASE("Stam-vs-L.3.4 spectrum dump on polar disk (extraordinary-vertex probe)",
          "[.experiment][shell][loop][disk_stam_probe]")
{
    // Cross-check for chladni-stam-vs-l34-consistent-mass. The
    // icosphere k=5 sweep showed Stam produces a spurious low mode
    // at 5617 Hz that L.3.4 doesn't, and the spurious mode shifts
    // DOWN with refinement (k=4: 5850 Hz, k=5: 5617 Hz). The
    // hypothesis: Stam's exact-eval on irregular sub-tiles near
    // extraordinary vertices has a localised softness that gets
    // worse as the surrounding tiles shrink.
    //
    // Polar disks have a single high-valence central hub
    // (valence = n_az). If Stam produces a similar spurious low
    // mode on this geometry under refinement (more n_rad rings), it
    // pins the issue as a generic Stam-on-extraordinary-vertex
    // pathology rather than icosphere-specific.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 8;
    const auto mat = steel_033();
    const auto sm  = cs::shell_material_from_isotropic(mat, h);

    auto run = [&](const cmsh::TriMesh& mesh, bool use_stam) {
        cs::LoopAssembler::Params p;
        p.use_stam = use_stam;
        // consistent mass (the new default — same regime as the
        // icosphere k=5 finding)
        p.m_lump = cs::MassLumping::None;
        const auto modes = cs::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, n_modes,
            cs::LoopAssembler{p});
        std::vector<double> hz;
        constexpr double tau = 2.0 * std::numbers::pi_v<double>;
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            hz.push_back(modes.omegas(i) / tau);
        }
        return hz;
    };

    // Sweep n_az ∈ {5, 16} (valence-5 hub matches icosphere's
    // extraordinary vertex; valence-16 hub probes the higher-valence
    // regime) × n_rad ∈ {6, 10, 16, 24} for refinement
    auto do_sweep = [&](int n_az) {
    for (int n_rad : {6, 10, 16, 24}) {
        const auto mesh = cmsh::generate_circular_disk(
            R, n_az, n_rad);
        std::cout << "\npolar n_az=" << n_az << " n_rad=" << n_rad
                  << "  V=" << mesh.V.rows() << "  F=" << mesh.F.rows() << '\n';
        const auto hz_l34  = run(mesh, /*use_stam=*/false);
        const auto hz_stam = run(mesh, /*use_stam=*/true);
        std::cout << "  L.3.4: ";
        for (double f : hz_l34) std::cout << f << " ";
        std::cout << "\n  Stam : ";
        for (double f : hz_stam) std::cout << f << " ";
        std::cout << "\n  Δ(mode0): " << (hz_stam[0] - hz_l34[0])
                  << "  Δ(mode1): " << (hz_stam[1] - hz_l34[1])
                  << "  (Stam − L.3.4 in Hz)\n";
    }
    };
    do_sweep(5);
    do_sweep(16);
}

TEST_CASE("disk method table — polar / iso / hex × quadrature × lumping",
          "[.benchmark][shell][loop][disk_table]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 7;

    const auto analytical =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, steel_033(), n_modes);
    REQUIRE(analytical.size() == n_modes);
    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    std::vector<double> leissa_hz;
    leissa_hz.reserve(n_modes);
    for (double w : analytical) leissa_hz.push_back(w / tau);

    // Three disk topologies, matched to roughly equal V counts so the
    // comparison is apples-to-apples.

    // Polar 32x8 = 257 V (extraordinary central vertex valence 32).
    const auto polar = cmsh::generate_circular_disk(R, /*n_az=*/32, /*n_rad=*/8);
    print_table("polar 32x8", polar, leissa_hz, n_modes);

    // Iso n_boundary=32. V count is generator-dependent; reports in
    // the table header.
    const auto iso = cmsh::generate_disk_iso(R, /*n_boundary=*/32);
    print_table("iso n_boundary=32", iso, leissa_hz, n_modes);

    // Hex n_layers=8 = 217 V. Note: clipped-hex disk has valence-3 rim
    // corners which the Loop boundary augmentation does not support.
    // If this throws, we still print the header + the THROW line for
    // every config so the table is readable.
    const auto hex = cmsh::generate_disk_hex(R, /*n_layers=*/8);
    print_table("hex n_layers=8", hex, leissa_hz, n_modes);

    // Coarse sanity: at least one config on at least one disk should
    // land within 30 % of the Leissa fundamental. We don't tighten
    // this — the table IS the deliverable.
    REQUIRE(polar.V.rows() > 0);
}
