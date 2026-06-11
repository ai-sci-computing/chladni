/**
 * @file test_modes_vs_free_free_cylinder_analytic.cpp
 * @brief Block W.4 gating tests: FEM vs analytic, free-free shell.
 *
 * Two siblings exercising the CST + Wardetzky-IBM formulation. Both are
 * currently *gated off* (`[.skip]`) and serve as documentation of the
 * absolute-frequency calibration gap discovered after the W.2/W.3
 * implementation landed:
 *
 *   - **flat strip vs Euler-Bernoulli beam**: free-free flat
 *     rectangular strip mesh; the lowest beam mode comes out **~62%
 *     too high** (FEM 52.9 rad/s vs analytic 32.6 rad/s) at
 *     L=1m / W=5cm / h=1mm steel, 100×5 cells. The strip is flat,
 *     so CST has no curvature locking — the discrepancy is the IBM
 *     bending prefactor on this regular grid.
 *
 *   - **cylinder vs Rayleigh-Love inextensional**: free-free polygonal
 *     cylinder mesh; FEM ovalling ~60% over analytic at L/R=20
 *     (n_around=128). Combines the IBM calibration gap with CST
 *     locking on coarse polygonal cylinders.
 *
 * The mismatch traces to the discrete IBM Hessian per hinge,
 * @f$ K_e = (3 D / A_0) c c^\top \otimes I_3 @f$, being
 * direction-dependent on regular grids: for purely y-aligned bending
 * (w = y²) the per-hinge energy sums to ~2.4× the continuum
 * @f$(D/2)\int (\nabla^2 w)^2 dA@f$; for the isotropic Laplacian
 * (w = x²+y²) it's ~1.34×. The Wardetzky-2007 paper claims equivalence
 * to Kirchhoff-Love bending only for *isometric* deformations; the
 * over-stiffness in the modal eigenvalue problem reflects the deviation
 * from that constraint on a discrete grid.
 *
 * Resolving this needs either (a) a different per-hinge prefactor
 * derived from the continuum operator on the specific triangulation,
 * (b) a co-rotational dihedral formulation (Bergou 2008 / DER family),
 * or (c) higher-order shell elements. Tracked separately; do not block
 * the rest of Block W on this.
 */

#include <chladni/analytical/beam.hpp>
#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>
#include <chladni/shell/lme.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

namespace {

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

/// Build a flat rectangular strip mesh in the x-y plane with the
/// (a,b,c)+(a,c,d) triangulation: each cell's diagonal goes from
/// bottom-left to top-right (matches the test_geometry unit-square hinge).
chladni::mesh::TriMesh build_strip(double L, double W, int nx, int ny)
{
    chladni::mesh::TriMesh m;
    const Eigen::Index n_verts = (nx + 1) * (ny + 1);
    m.V.resize(n_verts, 3);
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const Eigen::Index k = j * (nx + 1) + i;
            m.V(k, 0) = L * static_cast<double>(i) / nx;
            m.V(k, 1) = W * static_cast<double>(j) / ny;
            m.V(k, 2) = 0.0;
        }
    }
    m.F.resize(2 * nx * ny, 3);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Eigen::Index a = j       * (nx + 1) + i;
            const Eigen::Index b = j       * (nx + 1) + (i + 1);
            const Eigen::Index c = (j + 1) * (nx + 1) + (i + 1);
            const Eigen::Index d = (j + 1) * (nx + 1) + i;
            const Eigen::Index t = 2 * (j * nx + i);
            m.F(t,     0) = a; m.F(t,     1) = b; m.F(t,     2) = c;
            m.F(t + 1, 0) = a; m.F(t + 1, 1) = c; m.F(t + 1, 2) = d;
        }
    }
    return m;
}

}  // namespace

TEST_CASE("free-free flat strip (Loop FEM): lowest 2 modes match Euler-Bernoulli beam",
          "[shell][modes][strip][validation][beam][loop]")
{
    // L.5c.6 unblock: this is the second of the two original
    // analytic-mismatch tests that triggered the Loop rewrite (the
    // other being the cylinder ovalling test, validated by L.7). The
    // legacy CST + IBM path (the [.skip]ed sibling below) gave 53 rad/s
    // vs analytic 32.6 — 62% over. Loop's strict C^1 basis should
    // converge cleanly because the strip is essentially a thin beam:
    // the first two free-free transverse-bending modes match the
    // Euler-Bernoulli closed form within FEM mesh resolution.
    //
    // The strip mesh has valence-3 boundary corners at two opposite
    // diagonal corners and valence-2 corners at the other two
    // (build_strip uses the (a,b,c)+(a,c,d) triangulation). All are
    // handled by the L.5c boundary augmentation. Interior vertices
    // are all valence-6 so the L.3.4 subdivision path does not fire.
    constexpr double L = 1.00;
    constexpr double W = 0.05;
    constexpr double h = 1.00e-3;
    constexpr int nx = 100;
    constexpr int ny = 5;

    const auto mesh = build_strip(L, W, nx, ny);

    constexpr std::size_t n_modes = 2;
    const chladni::analytical::RectangularBeam beam_geom{
        .length = L, .width = W, .thickness = h};
    const auto omegas_analytic = chladni::analytical::
        free_free_beam_angular_frequencies(beam_geom, steel(), n_modes);

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n_modes);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));

    for (std::size_t k = 0; k < n_modes; ++k) {
        const double w_fem = modes.omegas(static_cast<Eigen::Index>(k));
        const double w_an  = omegas_analytic[k];
        const double rel_err = std::abs(w_fem - w_an) / w_an;
        INFO("mode " << k
             << "  FEM = "     << w_fem << " rad/s"
             << "  analytic = " << w_an << " rad/s"
             << "  rel_err = "  << rel_err);
        REQUIRE(rel_err < 0.05);
    }
}

TEST_CASE("free-free flat strip: lowest 2 modes match Euler-Bernoulli beam (skipped — IBM calibration)",
          "[shell][modes][strip][validation][beam][.skip]")
{
    constexpr double L = 1.00;
    constexpr double W = 0.05;
    constexpr double h = 1.00e-3;
    constexpr int nx = 100;
    constexpr int ny = 5;

    const auto mesh = build_strip(L, W, nx, ny);

    constexpr std::size_t n_modes = 2;
    const chladni::analytical::RectangularBeam beam_geom{
        .length = L, .width = W, .thickness = h};
    const auto omegas_analytic = chladni::analytical::
        free_free_beam_angular_frequencies(beam_geom, steel(), n_modes);

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h, n_modes);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));

    for (std::size_t k = 0; k < n_modes; ++k) {
        const double w_fem = modes.omegas(static_cast<Eigen::Index>(k));
        const double w_an  = omegas_analytic[k];
        const double rel_err = std::abs(w_fem - w_an) / w_an;
        INFO("mode " << k
             << "  FEM = "     << w_fem << " rad/s"
             << "  analytic = " << w_an << " rad/s"
             << "  rel_err = "  << rel_err);
        REQUIRE(rel_err < 0.05);
    }
}

TEST_CASE("free-free cylinder: FEM ovalling matches Rayleigh-Love (skipped — IBM calibration + CST locking)",
          "[shell][modes][cylinder][validation][analytic][.skip]")
{
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    constexpr int n_around = 32;
    constexpr int n_along  = 80;

    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};

    constexpr std::size_t n_analytic = 4;
    const auto omegas_analytic = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, steel(), n_analytic);

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h, 2 * n_analytic);

    REQUIRE(modes.omegas.size() ==
            static_cast<Eigen::Index>(2 * n_analytic));

    for (std::size_t k = 0; k < n_analytic; ++k) {
        const double w_fem_a = modes.omegas(static_cast<Eigen::Index>(2 * k + 0));
        const double w_fem_b = modes.omegas(static_cast<Eigen::Index>(2 * k + 1));
        const double w_fem   = 0.5 * (w_fem_a + w_fem_b);
        const double w_an    = omegas_analytic[k];
        const double rel_err = std::abs(w_fem - w_an) / w_an;
        INFO("n = " << (k + 2)
             << "  FEM(a) = " << w_fem_a << " rad/s"
             << "  FEM(b) = " << w_fem_b << " rad/s"
             << "  analytic = " << w_an << " rad/s"
             << "  rel_err = " << rel_err);
        REQUIRE(rel_err < 0.10);
    }
}

namespace {

/// Per-resolution Loop FEM result for the free-free cylinder ovalling
/// validation. Used by both the single-resolution L.7 regression test
/// and the convergence test below.
struct CylinderOvallingResult {
    std::vector<double> fem_omegas;       ///< lowest-omega FEM mode per n in [2, 2 + n_circ)
    std::vector<double> analytic_omegas;  ///< Rayleigh-Love at the same n's
    std::vector<double> rel_errs;
};

CylinderOvallingResult run_free_free_cylinder_at(
    double R, double L, double h,
    int n_around, int n_along,
    std::size_t n_circumferential,
    std::size_t n_modes_solve)
{
    const auto mesh =
        chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const auto omegas_analytic = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, steel(), n_circumferential);

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n_modes_solve);

    // Fourier-basis projection of the radial displacement onto each
    // circumferential wavenumber, summed over rings. Returns the
    // dominant n in [0, n_max] for each mode.
    constexpr int n_max = 6;
    const double  two_pi = 2.0 * std::numbers::pi_v<double>;
    auto dominant_n = [&](const Eigen::VectorXd& u)
    {
        std::array<double, n_max + 1> Es{};
        for (int n = 0; n <= n_max; ++n) {
            double total = 0.0;
            for (int j = 0; j <= n_along; ++j) {
                double c = 0.0, s = 0.0;
                for (int i = 0; i < n_around; ++i) {
                    const Eigen::Index v = j * n_around + i;
                    const double phi = two_pi * i / n_around;
                    const double cos_phi = std::cos(phi);
                    const double sin_phi = std::sin(phi);
                    const double ux = u(3 * v + 0);
                    const double uy = u(3 * v + 1);
                    const double u_r = ux * cos_phi + uy * sin_phi;
                    c += u_r * std::cos(n * phi);
                    s += u_r * std::sin(n * phi);
                }
                c *= 2.0 / n_around;
                s *= 2.0 / n_around;
                total += c * c + s * s;
            }
            Es[static_cast<std::size_t>(n)] = total;
        }
        int best = 0;
        for (int n = 1; n <= n_max; ++n) {
            if (Es[static_cast<std::size_t>(n)]
                > Es[static_cast<std::size_t>(best)]) best = n;
        }
        return best;
    };

    // Find the lowest-omega FEM mode at each circumferential n in
    // [2, 2 + n_circumferential).
    std::vector<double> omega_n_fem(n_circumferential, -1.0);
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        const int n = dominant_n(modes.shapes.col(k));
        if (n < 2 || n > 1 + static_cast<int>(n_circumferential)) continue;
        const std::size_t slot = static_cast<std::size_t>(n - 2);
        if (omega_n_fem[slot] < 0.0) {
            omega_n_fem[slot] = modes.omegas(k);
        }
    }

    CylinderOvallingResult r;
    r.fem_omegas      = std::move(omega_n_fem);
    r.analytic_omegas.assign(omegas_analytic.begin(),
                              omegas_analytic.end());
    r.rel_errs.reserve(n_circumferential);
    for (std::size_t k = 0; k < n_circumferential; ++k) {
        if (r.fem_omegas[k] <= 0.0) {
            r.rel_errs.push_back(-1.0);
        } else {
            r.rel_errs.push_back(
                std::abs(r.fem_omegas[k] - r.analytic_omegas[k])
                / r.analytic_omegas[k]);
        }
    }
    return r;
}

}  // namespace

TEST_CASE("free-free cylinder (Loop FEM): FEM ovalling matches Rayleigh-Love",
          "[shell][modes][cylinder][validation][analytic][loop]")
{
    // L.7 validation: the Cirak-Ortiz Loop-subdivision shell drives the
    // free-free cylinder's inextensional ovalling spectrum. The legacy
    // CST + IBM path (the [.skip]ed sibling above) over-stiffens by ~60%
    // on this geometry; Loop's strict C^1 box-spline basis converges to
    // within a few percent of Rayleigh-Love.
    //
    // Mode-shape filter: the eigensolver returns ALL low-frequency modes
    // for each circumferential wavenumber n — including n=2 modes with
    // 0, 1, 2, ... axial half-waves before reaching the n=3 ovalling
    // pair, plus n=1 lateral and n=0 breathing/extensional (filtered).
    // The inextensional ovalling reference is the LOWEST-omega mode at
    // each n, so we project the radial displacement of every returned
    // mode onto the Fourier basis cos(n φ) / sin(n φ), pick the
    // dominant n per mode, and take the smallest-omega pair per n.
    //
    // Diagnostic runs on n_around=24 n_along=32:
    //   Lumped (pre-2026-05-17 default):
    //     n=2: 415 rad/s vs 410 (+1.2 %),  n=3: 1150 vs 1159 (-0.8 %),
    //     n=4: 2120 vs 2223 (-4.6 %).
    //   Consistent (post-2026-05-17 default):
    //     n=2: ~430 rad/s (+4.5 %), n=3: 1256 (+8.4 %),
    //     n=4: 2517 (+13.3 %).
    // The flip-sign-and-magnitude shift confirms the textbook
    // expectation that lumped mass lowers eigenfrequencies broadly
    // (lumping happened to bias the FEM toward the Rayleigh-Love
    // analytic — coincidence, not correctness). Under consistent
    // mass the FEM overshoots Rayleigh-Love by ~5-13 % at this
    // resolution, with the gap widening on the higher-n modes that
    // are most circumferentially under-resolved (24 around = 6 per
    // lobe at n=4). The convergence test below pins the trajectory
    // shrinks geometrically with mesh refinement, so this is a
    // genuine "FEM converges to its own (higher) Kirchhoff-Love
    // limit, while Rayleigh-Love itself sits ~9 % below true K-L".
    const auto r = run_free_free_cylinder_at(
        /*R=*/0.10, /*L=*/2.00, /*h=*/1.00e-3,
        /*n_around=*/24, /*n_along=*/32,
        /*n_circumferential=*/3,
        /*n_modes_solve=*/30);

    for (std::size_t k = 0; k < r.fem_omegas.size(); ++k) {
        INFO("n = " << (k + 2)
             << "  FEM = "      << r.fem_omegas[k] << " rad/s"
             << "  analytic = " << r.analytic_omegas[k] << " rad/s"
             << "  rel_err = "  << r.rel_errs[k]);
        REQUIRE(r.fem_omegas[k] > 0.0);
        REQUIRE(r.rel_errs[k] < 0.15);
    }
}

TEST_CASE("free-free cylinder: 4-algorithm frequency comparison",
          "[.diag][shell][modes][cylinder][algo_compare]")
{
    // Diagnostic: why do the 4 GUI algorithms (Loop, LME-1st, SME-2nd,
    // Legacy CST+IBM) agree on the sphere but spread widely on the
    // cylinder? Prints the lowest physical (rigid-filtered) frequencies
    // each produces on the SAME free-free cylinder, against the
    // Rayleigh inextensional-ovalling reference. The cylinder's low
    // modes are inextensional bending — the canonical thin-shell
    // membrane-locking stress test — so the spread measures how each
    // formulation handles bending/membrane separation on a developable
    // (zero-Gaussian-curvature) surface. A closed sphere admits almost
    // no inextensional modes, which is why all four agree there.
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    constexpr int    n_around = 24;
    constexpr int    n_along  = 32;
    constexpr std::size_t n_modes = 6;

    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const auto mat  = steel();
    const auto sm   = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const auto an = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, mat, 4);

    std::fprintf(stderr,
        "[cyl_algo] free-free cylinder R=%.2f L=%.2f h=%.0e  "
        "%dx%d (%lld V)\n", R, L, h, n_around, n_along,
        static_cast<long long>(mesh.V.rows()));
    std::fprintf(stderr,
        "[cyl_algo] Rayleigh inextensional ovalling: "
        "n=2 %.1f  n=3 %.1f  n=4 %.1f  rad/s\n",
        an[0], an[1], an[2]);

    using chladni::shell::compute_shell_modes;
    // Each algorithm is run under try/catch: LME locks hard and SME's
    // in-chart Newton diverges on this developable geometry, both now
    // surfacing as catchable std::exceptions (see the run_threaded fix).
    auto report = [&](const char* name, auto&& solve) {
        try {
            const chladni::shell::ShellModes m = solve();
            std::fprintf(stderr, "[cyl_algo] %-14s omegas:", name);
            for (Eigen::Index k = 0; k < m.omegas.size(); ++k) {
                std::fprintf(stderr, " %.1f", m.omegas(k));
            }
            if (m.omegas.size() >= 2) {
                const double w_n2 = 0.5 * (m.omegas(0) + m.omegas(1));
                std::fprintf(stderr,
                    "  | n=2 mean %.1f (vs Rayleigh %+.1f%%)",
                    w_n2, 100.0 * (w_n2 - an[0]) / an[0]);
            }
            std::fprintf(stderr, "\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[cyl_algo] %-14s DIVERGED — %s\n",
                name, e.what());
        }
    };

    report("Loop", [&] {
        chladni::shell::LoopAssembler a;
        return compute_shell_modes(mesh.V, mesh.F, mat, sm, h, n_modes, a);
    });
    report("LME-1st", [&] {
        chladni::shell::LMEAssembler a{chladni::shell::LMEAssembler::Params{}};
        return compute_shell_modes(mesh.V, mesh.F, mat, sm, h, n_modes, a);
    });
    // Audit lead: Millán 2011 fixes γ_LME = 0.8 and warns γ ≥ 1.0
    // degrades convergence; our default is 1.6. Sweep γ to see whether
    // the +239 % locking is (partly) the γ deviation.
    for (double g : {0.4, 0.6, 0.8, 1.0}) {
        report((std::string("LME γ=") + std::to_string(g)).c_str(), [&] {
            chladni::shell::LMEAssembler::Params p;
            p.gamma = g;
            chladni::shell::LMEAssembler a{p};
            return compute_shell_modes(mesh.V, mesh.F, mat, sm, h, n_modes, a);
        });
    }
    report("SME-2nd", [&] {
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        chladni::shell::LMEAssembler a{p};
        return compute_shell_modes(mesh.V, mesh.F, mat, sm, h, n_modes, a);
    });
    report("Legacy", [&] {
        return compute_shell_modes(mesh.V, mesh.F, mat, h, n_modes);
    });
    SUCCEED("4-algorithm cylinder comparison printed to stderr");
}

TEST_CASE("LME cylinder locking — refinement + membrane/bending split",
          "[.diag][shell][modes][cylinder][lme_locking]")
{
    // Bug hunt for the +239 % LME cylinder over-stiffness.
    //   (1) Refinement: locking does NOT improve with h; under-
    //       resolution drops ~O(h^2). Run LME n=2 ovalling at 3 levels.
    //   (2) Energy split: take the lowest LME mode's shape and measure
    //       its membrane vs bending energy via assemble_K with k_L=0
    //       (bending only) and k_B=0 (membrane only). A true
    //       inextensional ovalling mode has E_membrane << E_bending;
    //       parasitic-membrane locking shows E_membrane >> E_bending.
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const double rayleigh_n2 =
        chladni::analytical::free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, mat, 1)[0];

    std::fprintf(stderr, "[lme_lock] Rayleigh n=2 = %.1f rad/s\n", rayleigh_n2);

    // Dominant circumferential wavenumber of a mode's radial field, plus
    // the FRACTION of radial energy in that n (to flag spurious modes
    // whose "n=2" projection is weak/noisy rather than clean ovalling).
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    auto n_spectrum = [&](const Eigen::VectorXd& u, int na, int nl) {
        std::array<double, 7> E{};
        double total = 0.0;
        for (int n = 0; n <= 6; ++n) {
            double tot = 0.0;
            for (int j = 0; j <= nl; ++j) {
                double c = 0, s = 0;
                for (int i = 0; i < na; ++i) {
                    const Eigen::Index v = j * na + i;
                    if (3 * v + 1 >= u.size()) continue;
                    const double phi = two_pi * i / na;
                    const double ur = u(3*v+0)*std::cos(phi) + u(3*v+1)*std::sin(phi);
                    c += ur * std::cos(n * phi); s += ur * std::sin(n * phi);
                }
                tot += c*c + s*s;
            }
            E[static_cast<std::size_t>(n)] = tot; total += tot;
        }
        int best = 0;
        for (int n = 1; n <= 6; ++n)
            if (E[static_cast<std::size_t>(n)] > E[static_cast<std::size_t>(best)]) best = n;
        const double frac = total > 0 ? E[static_cast<std::size_t>(best)] / total : 0.0;
        return std::pair<int, double>{best, frac};
    };
    struct Res { int na; int nl; };
    for (Res r : {Res{16, 24}, Res{24, 32}, Res{32, 48},
                  Res{48, 64}, Res{64, 96}, Res{96, 128}}) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, r.na, r.nl);
        chladni::shell::LMEAssembler a{chladni::shell::LMEAssembler::Params{}};
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 12, a);
        // lowest mode with dominant n==2
        double w_n2 = -1.0; int k_n2 = -1; double frac_n2 = 0.0;
        for (Eigen::Index k = 0; k < m.omegas.size(); ++k) {
            const auto [n, frac] = n_spectrum(m.shapes.col(k), r.na, r.nl);
            if (n == 2) { w_n2 = m.omegas(k); k_n2 = static_cast<int>(k); frac_n2 = frac; break; }
        }
        std::fprintf(stderr,
            "[lme_lock] refine %dx%d (%lld V): n=2 ovalling LME %.1f  "
            "(vs Rayleigh %+.0f%%)  [mode #%d, n=2 energy frac %.2f]\n",
            r.na, r.nl, static_cast<long long>(mesh.V.rows()),
            w_n2, 100.0 * (w_n2 - rayleigh_n2) / rayleigh_n2, k_n2, frac_n2);
        if (r.na >= 48) {   // characterise the low spectrum at fine mesh
            for (Eigen::Index k = 0; k < std::min<Eigen::Index>(8, m.omegas.size()); ++k) {
                const auto [n, frac] = n_spectrum(m.shapes.col(k), r.na, r.nl);
                std::fprintf(stderr,
                    "[lme_lock]     mode #%lld  w=%.1f  dominant n=%d (frac %.2f)\n",
                    static_cast<long long>(k), m.omegas(k), n, frac);
            }
        }
    }

    // Support-size sweep at fixed mesh: does WIDER meshfree support
    // relieve membrane locking (more overlap → smoother inextensional
    // representation)? The live support-width knob on the curved path
    // is γ (lower γ → larger value-based radius h_a·sqrt(ln(1/tol)/γ)
    // AND a flatter basis); r_cut_mult_curved is retired/inert (the
    // cutoff is value-based via tol_lme), so it must NOT be swept here.
    {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, 24, 32);
        // Two modest widths: γ=1.6 (default, ~3.8h) and γ=1.0 (~4.8h).
        // Lower γ widens the support but the per-Gauss Newton cost grows
        // fast (γ<0.8 → O(n²) active sets); two points show the trend
        // direction without blowing up this already-slow [.diag].
        for (double g : {1.6, 1.0}) {
            chladni::shell::LMEAssembler::Params p;
            p.gamma = g;
            try {
                chladni::shell::LMEAssembler a{p};
                const auto m = chladni::shell::compute_shell_modes(
                    mesh.V, mesh.F, mat, sm, h, 4, a);
                const double w = 0.5 * (m.omegas(0) + m.omegas(1));
                std::fprintf(stderr,
                    "[lme_lock] gamma=%.1f (24x32, wider as gamma↓): n=2 "
                    "LME %.1f  (vs Rayleigh %+.0f%%)\n",
                    g, w, 100.0 * (w - rayleigh_n2) / rayleigh_n2);
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "[lme_lock] gamma=%.1f DIVERGED — %s\n", g, e.what());
            }
        }
    }

    // Energy split on the lowest ovalling mode — ghost-free so the
    // mode shape (3*n_v) matches the assembled K (3*n_v). For a TRUE
    // inextensional ovalling mode E_membrane << E_bending; parasitic-
    // membrane locking inverts this. Contrast LME vs Loop on the SAME
    // mesh: Loop is known-good (+4.6%), so its split is the reference.
    const auto mesh = chladni::mesh::generate_cylinder(R, L, 24, 32);
    chladni::shell::ShellMaterial sm_mem{.k_L = sm.k_L, .k_B = 0.0};
    chladni::shell::ShellMaterial sm_bend{.k_L = 0.0, .k_B = sm.k_B};

    auto split = [&](const char* name, const chladni::shell::ShellModes& m,
                     const Eigen::SparseMatrix<double>& K_mem,
                     const Eigen::SparseMatrix<double>& K_bend) {
        const Eigen::VectorXd u = m.shapes.col(0);
        if (u.size() != K_mem.rows()) {
            std::fprintf(stderr, "[lme_lock] %s: dof mismatch %lld vs %lld\n",
                name, static_cast<long long>(u.size()),
                static_cast<long long>(K_mem.rows()));
            return;
        }
        const double e_mem  = 0.5 * u.dot(K_mem  * u);
        const double e_bend = 0.5 * u.dot(K_bend * u);
        std::fprintf(stderr,
            "[lme_lock] %-5s lowest mode (w=%.1f): E_mem %.3e  E_bend %.3e  "
            "E_mem/E_bend = %.3f\n",
            name, m.omegas(0), e_mem, e_bend, e_mem / e_bend);
    };

    {   // LME, ghost-free
        chladni::shell::LMEAssembler::Params p;
        p.use_ghost_nodes = false;
        chladni::shell::LMEAssembler a{p};
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 4, a);
        chladni::shell::LMEAssembler a_m{p}, a_b{p};
        split("LME", m, a_m.assemble_K(mesh.V, mesh.F, sm_mem),
                        a_b.assemble_K(mesh.V, mesh.F, sm_bend));
    }
    {   // Loop reference
        chladni::shell::LoopAssembler a, a_m, a_b;
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 4, a);
        split("Loop", m, a_m.assemble_K(mesh.V, mesh.F, sm_mem),
                         a_b.assemble_K(mesh.V, mesh.F, sm_bend));
    }

    // PRESCRIBED-FIELD TEST (removes the eigensolver from the loop):
    // hand both methods the EXACT analytic n=2 ring-ovalling field,
    // which is inextensional by construction (u_r=cos2θ, u_θ=-½sin2θ,
    // u_z=0 ⇒ ε_θθ = (u_θ,θ+u_r)/R = 0, ε_zz=ε_θz=0). The CONTINUOUS
    // membrane energy is exactly zero; a faithful discrete membrane
    // operator gives ≈0. A large value means that method's discrete
    // membrane STRAIN is spurious on the curved chart (fidelity bug),
    // not just an eigensolver artifact.
    Eigen::VectorXd u_inext(3 * mesh.V.rows());
    for (Eigen::Index v = 0; v < mesh.V.rows(); ++v) {
        const double x = mesh.V(v, 0), y = mesh.V(v, 1);
        const double th = std::atan2(y, x);
        const double u_r =  std::cos(2.0 * th);
        const double u_t = -0.5 * std::sin(2.0 * th);
        u_inext(3 * v + 0) = u_r * std::cos(th) - u_t * std::sin(th);
        u_inext(3 * v + 1) = u_r * std::sin(th) + u_t * std::cos(th);
        u_inext(3 * v + 2) = 0.0;
    }
    auto field_energy = [&](const char* name,
                            const Eigen::SparseMatrix<double>& K_mem,
                            const Eigen::SparseMatrix<double>& K_bend) {
        const double e_mem  = 0.5 * u_inext.dot(K_mem  * u_inext);
        const double e_bend = 0.5 * u_inext.dot(K_bend * u_inext);
        std::fprintf(stderr,
            "[lme_lock] prescribed inextensional field, %-5s: "
            "E_mem %.3e  E_bend %.3e  E_mem/E_bend = %.3f\n",
            name, e_mem, e_bend, e_mem / e_bend);
    };
    {
        chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
        chladni::shell::LMEAssembler a_m{p}, a_b{p};
        field_energy("LME", a_m.assemble_K(mesh.V, mesh.F, sm_mem),
                            a_b.assemble_K(mesh.V, mesh.F, sm_bend));
    }
    {
        chladni::shell::LoopAssembler a_m, a_b;
        field_energy("Loop", a_m.assemble_K(mesh.V, mesh.F, sm_mem),
                             a_b.assemble_K(mesh.V, mesh.F, sm_bend));
    }
    SUCCEED("LME cylinder locking diagnostic printed to stderr");
}

TEST_CASE("LME cylinder bending — prescribed-field convergence (CONFOUNDED)",
          "[.diag][shell][modes][cylinder][lme_bend_conv]")
{
    // CAVEAT — this test was BIASED against LME and the apparent
    // "bending deficit" turned out to be a CONFOUND, not the bug. Kept
    // as a permanent reminder + counterpart to `[lme_bend_flat]`.
    //
    // We feed the EXACT analytic n=2 inextensional field (u_r=cos2θ,
    // u_θ=-½sin2θ, u_z=0; κ_θθ=(1-n²)/R²cos(nθ)) into K_bend and compare
    // uᵀK_bend u/2 to the continuum bending energy
    //   E_bend_an = (9π D L)/(2 R³),   D = E h³/(12(1-ν²)).
    // Loop converges cleanly to 1.0× analytic (0.88 → 0.97 → 0.99 at
    // 24x32 → 48x64 → 96x128). LME PLATEAUS at ~0.62×.
    //
    // The 0.62 plateau is NOT a bending-operator bug — `[lme_bend_flat]`
    // shows LME ALSO gives ~0.53× on a FLAT plate, yet flat modal
    // frequencies are accurate to <1 %. LME is a 1st-order-consistent
    // *approximant*; it smooths the prescribed nodal field, so
    // ‖∇²u_LME‖ < ‖∇²u_analytic‖. Modal analysis self-consistently uses
    // the smoothed field on BOTH sides of the Rayleigh quotient and is
    // therefore unaffected. The real bug ([loop_vs_lme_refine] +
    // [cross_operator]) is a SPURIOUS soft mode the LME stiffness admits
    // on the cylinder, NOT a uniform bending under-computation.
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    chladni::shell::ShellMaterial sm_bend{.k_L = 0.0, .k_B = sm.k_B};

    const double E_an = 200.0e9, nu = 0.30;
    const double D = E_an * h * h * h / (12.0 * (1.0 - nu * nu));
    const double e_bend_analytic =
        9.0 * std::numbers::pi_v<double> * D * L / (2.0 * R * R * R);
    std::fprintf(stderr,
        "[lme_bend] analytic E_bend(unit n=2 inextensional) = %.4e  "
        "(D=%.4f)\n", e_bend_analytic, D);

    struct Res { int na; int nl; };
    for (Res r : {Res{24, 32}, Res{48, 64}, Res{96, 128}}) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, r.na, r.nl);
        Eigen::VectorXd u(3 * mesh.V.rows());
        for (Eigen::Index v = 0; v < mesh.V.rows(); ++v) {
            const double x = mesh.V(v, 0), y = mesh.V(v, 1);
            const double th = std::atan2(y, x);
            const double u_r =  std::cos(2.0 * th);
            const double u_t = -0.5 * std::sin(2.0 * th);
            u(3 * v + 0) = u_r * std::cos(th) - u_t * std::sin(th);
            u(3 * v + 1) = u_r * std::sin(th) + u_t * std::cos(th);
            u(3 * v + 2) = 0.0;
        }
        double e_lme = 0.0, e_loop = 0.0;
        {
            chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
            chladni::shell::LMEAssembler a{p};
            const auto K = a.assemble_K(mesh.V, mesh.F, sm_bend);
            e_lme = 0.5 * u.dot(K * u);
        }
        {
            chladni::shell::LoopAssembler a;
            const auto K = a.assemble_K(mesh.V, mesh.F, sm_bend);
            e_loop = 0.5 * u.dot(K * u);
        }
        const double inv = 1.0 / e_bend_analytic;
        std::fprintf(stderr,
            "[lme_bend] %dx%d (%lld V): LME=%.3e (%.3f× analytic)  "
            "Loop=%.3e (%.3f×)\n",
            r.na, r.nl, static_cast<long long>(mesh.V.rows()),
            e_lme, e_lme * inv, e_loop, e_loop * inv);
    }
    SUCCEED("LME bending-convergence diagnostic printed to stderr");
}

TEST_CASE("LME bending operator — flat-plate prescribed-field control + "
          "cylinder γ sweep",
          "[.diag][shell][modes][lme_bend_flat]")
{
    const auto mat = steel();
    constexpr double h = 1.00e-3;
    const auto sm = chladni::shell::shell_material_from_isotropic(mat, h);
    chladni::shell::ShellMaterial sm_bend{.k_L = 0.0, .k_B = sm.k_B};
    const double E_an = 200.0e9, nu = 0.30;
    const double D = E_an * h * h * h / (12.0 * (1.0 - nu * nu));

    // (1) FLAT control: strip L×W in xy, prescribe out-of-plane bending
    //     w(x) = cos(k x), k = 2π m / L (m full periods). Then
    //     ρ_xx = -w'' = k² cos(kx), ρ_yy=ρ_xy=0, and
    //     E_bend = (D/2) k⁴ ∫cos²(kx) dA = (D/2) k⁴ (L/2) W.
    std::fprintf(stderr, "[bend_flat] --- flat strip control ---\n");
    for (int nx : {64, 128, 256}) {
        const double L = 1.0, W = 0.25;
        const int    ny = std::max(4, nx / 16);
        const auto   mesh = build_strip(L, W, nx, ny);
        const int    m_per = 2;
        const double k = 2.0 * std::numbers::pi_v<double> * m_per / L;
        Eigen::VectorXd u(3 * mesh.V.rows());
        u.setZero();
        for (Eigen::Index v = 0; v < mesh.V.rows(); ++v) {
            u(3 * v + 2) = std::cos(k * mesh.V(v, 0));   // w out-of-plane
        }
        const double e_an = 0.5 * D * std::pow(k, 4) * (L / 2.0) * W;
        double e_lme = 0.0, e_loop = 0.0;
        {
            chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
            chladni::shell::LMEAssembler a{p};
            e_lme = 0.5 * u.dot(a.assemble_K(mesh.V, mesh.F, sm_bend) * u);
        }
        {
            chladni::shell::LoopAssembler a;
            e_loop = 0.5 * u.dot(a.assemble_K(mesh.V, mesh.F, sm_bend) * u);
        }
        std::fprintf(stderr,
            "[bend_flat] strip nx=%d: analytic=%.4e  LME=%.4e (%.3f×)  "
            "Loop=%.4e (%.3f×)\n",
            nx, e_an, e_lme, e_lme / e_an, e_loop, e_loop / e_an);
    }

    // (2) Cylinder prescribed-field γ sweep at a FIXED fine mesh: does a
    //     SHARPER LME basis (larger γ → less Hessian smoothing) recover
    //     bending energy toward the analytic? If yes, the deficit is
    //     basis over-smoothing; if flat, it is geometric.
    std::fprintf(stderr, "[bend_flat] --- cylinder γ sweep (48x64) ---\n");
    {
        constexpr double R = 0.10, Lc = 2.00;
        const double e_bend_analytic =
            9.0 * std::numbers::pi_v<double> * D * Lc / (2.0 * R * R * R);
        const auto mesh = chladni::mesh::generate_cylinder(R, Lc, 48, 64);
        Eigen::VectorXd u(3 * mesh.V.rows());
        for (Eigen::Index v = 0; v < mesh.V.rows(); ++v) {
            const double th = std::atan2(mesh.V(v, 1), mesh.V(v, 0));
            const double u_r =  std::cos(2.0 * th);
            const double u_t = -0.5 * std::sin(2.0 * th);
            u(3 * v + 0) = u_r * std::cos(th) - u_t * std::sin(th);
            u(3 * v + 1) = u_r * std::sin(th) + u_t * std::cos(th);
            u(3 * v + 2) = 0.0;
        }
        for (double g : {0.8, 1.6, 3.2, 6.4, 12.8}) {
            chladni::shell::LMEAssembler::Params p;
            p.use_ghost_nodes = false;
            p.gamma = g;
            chladni::shell::LMEAssembler a{p};
            const double e = 0.5 * u.dot(
                a.assemble_K(mesh.V, mesh.F, sm_bend) * u);
            std::fprintf(stderr,
                "[bend_flat] cyl γ=%.1f: E_bend=%.4e (%.3f× analytic)\n",
                g, e, e / e_bend_analytic);
        }
    }
    SUCCEED("flat control + γ sweep printed to stderr");
}

TEST_CASE("cylinder n=2 ovalling — Loop vs LME refinement (does LOOP also "
          "soften?)",
          "[.diag][shell][modes][cylinder][loop_vs_lme_refine]")
{
    // THE decisive check the handoff flagged as never-run: refine BOTH
    // Loop and LME on the free-free cylinder n=2 ovalling and print them
    // side by side. If Loop stabilises near Rayleigh (~410-428) while LME
    // sinks past it, LME has a bug. If Loop ALSO sinks toward LME's
    // limit, then Rayleigh-inextensional is just a soft reference and
    // LME may be converging correctly (true thin-shell KL < Rayleigh).
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const double rayleigh_n2 = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, mat, 1)[0];
    std::fprintf(stderr, "[lvl] Rayleigh n=2 = %.1f rad/s\n", rayleigh_n2);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    auto dom_n = [&](const Eigen::VectorXd& u, int na, int nl) {
        std::array<double, 7> E{}; double tot = 0;
        for (int n = 0; n <= 6; ++n) {
            double t = 0;
            for (int j = 0; j <= nl; ++j) {
                double c = 0, s = 0;
                for (int i = 0; i < na; ++i) {
                    const Eigen::Index v = j * na + i;
                    if (3 * v + 1 >= u.size()) continue;
                    const double phi = two_pi * i / na;
                    const double ur = u(3*v+0)*std::cos(phi)
                                    + u(3*v+1)*std::sin(phi);
                    c += ur * std::cos(n*phi); s += ur * std::sin(n*phi);
                }
                t += c*c + s*s;
            }
            E[static_cast<std::size_t>(n)] = t; tot += t;
        }
        int best = 0;
        for (int n = 1; n <= 6; ++n)
            if (E[static_cast<std::size_t>(n)]
                > E[static_cast<std::size_t>(best)]) best = n;
        return best;
    };
    auto lowest_n2 = [&](const chladni::shell::ShellModes& m, int na, int nl) {
        for (Eigen::Index k = 0; k < m.omegas.size(); ++k)
            if (dom_n(m.shapes.col(k), na, nl) == 2) return m.omegas(k);
        return -1.0;
    };

    struct Res { int na; int nl; };
    for (Res r : {Res{24, 32}, Res{32, 48}, Res{48, 64}, Res{64, 96}}) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, r.na, r.nl);
        double w_loop = -1, w_lme = -1;
        {
            chladni::shell::LoopAssembler a;
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 12, a);
            w_loop = lowest_n2(m, r.na, r.nl);
        }
        {
            chladni::shell::LMEAssembler a{chladni::shell::LMEAssembler::Params{}};
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 12, a);
            w_lme = lowest_n2(m, r.na, r.nl);
        }
        std::fprintf(stderr,
            "[lvl] %dx%d (%lld V): Loop n=2 %.1f (%+.0f%%)   "
            "LME n=2 %.1f (%+.0f%%)\n",
            r.na, r.nl, static_cast<long long>(mesh.V.rows()),
            w_loop, 100.0 * (w_loop - rayleigh_n2) / rayleigh_n2,
            w_lme, 100.0 * (w_lme - rayleigh_n2) / rayleigh_n2);
    }
    SUCCEED("Loop-vs-LME cylinder refinement printed to stderr");
}

TEST_CASE("cylinder n=2 — cross-operator Rayleigh quotient (K vs M isolation)",
          "[.diag][shell][modes][cylinder][cross_operator]")
{
    // Take the (too-soft) ghost-free LME n=2 eigenmode and evaluate it
    // under BOTH LME's and Loop's K and M. If u under Loop's operators
    // gives the physical ω, the LME mode SHAPE is fine and an LME
    // OPERATOR is wrong; comparing uᵀK_lme u vs uᵀK_loop u and
    // uᵀM_lme u vs uᵀM_loop u then says whether the stiffness (bending)
    // or the mass is the culprit. Do the symmetric thing for Loop's mode.
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    for (auto [na, nl] : {std::pair{48, 64}, std::pair{64, 96}}) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, na, nl);
        chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
        chladni::shell::LMEAssembler a_lme{p};
        chladni::shell::LoopAssembler a_loop;

        const double rho_h = mat.density * h;
        const auto K_lme  = a_lme.assemble_K(mesh.V, mesh.F, sm);
        const auto M_lme  = a_lme.assemble_M(mesh.V, mesh.F, rho_h);
        const auto K_loop = a_loop.assemble_K(mesh.V, mesh.F, sm);
        const auto M_loop = a_loop.assemble_M(mesh.V, mesh.F, rho_h);

        const auto m_lme = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 6, a_lme);
        const auto m_loop = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 6, a_loop);

        auto rq = [](const Eigen::SparseMatrix<double>& K,
                     const Eigen::SparseMatrix<double>& M,
                     const Eigen::VectorXd& u) {
            return std::sqrt((u.dot(K * u)) / (u.dot(M * u)));
        };
        const Eigen::VectorXd u_lme  = m_lme.shapes.col(0);
        const Eigen::VectorXd u_loop = m_loop.shapes.col(0);

        std::fprintf(stderr, "[xop] === %dx%d (%lld V) ===\n",
            na, nl, static_cast<long long>(mesh.V.rows()));
        std::fprintf(stderr,
            "[xop] LME  mode: ω(K_lme,M_lme)=%.1f  ω(K_loop,M_loop)=%.1f  "
            "| Kbend·: uKlme=%.3e uKloop=%.3e  uMlme=%.3e uMloop=%.3e\n",
            rq(K_lme, M_lme, u_lme), rq(K_loop, M_loop, u_lme),
            u_lme.dot(K_lme * u_lme),  u_lme.dot(K_loop * u_lme),
            u_lme.dot(M_lme * u_lme),  u_lme.dot(M_loop * u_lme));
        std::fprintf(stderr,
            "[xop] Loop mode: ω(K_lme,M_lme)=%.1f  ω(K_loop,M_loop)=%.1f  "
            "| uKlme=%.3e uKloop=%.3e  uMlme=%.3e uMloop=%.3e\n",
            rq(K_lme, M_lme, u_loop), rq(K_loop, M_loop, u_loop),
            u_loop.dot(K_lme * u_loop),  u_loop.dot(K_loop * u_loop),
            u_loop.dot(M_lme * u_loop),  u_loop.dot(M_loop * u_loop));
    }
    SUCCEED("cross-operator Rayleigh quotient printed to stderr");
}

TEST_CASE("cylinder LME basis CONSISTENCY at a Gauss point (numerical "
          "check of paper identities)",
          "[.diag][shell][cylinder][lme_basis_consistency]")
{
    // Distinguish "structural O(h^2) consistency error" from "real
    // implementation bug" in the LME basis derivatives on a curved
    // shell. The paper's 1st-order consistency identities are:
    //   (a)   Σ_b p_b(x) = 1
    //   (b)   Σ_b ∂_α p_b(x) = 0
    //   (c)   Σ_b p_b(x) · ξ_b = x         (linear reproduction)
    //   (d)   Σ_b ∂_α p_b(x) · ξ_b^β = δ_α^β  (gradient of (c))
    // If these hold to ~ulp on a cylinder patch, the LME basis itself
    // is faithful and the 7.7× spurious membrane on the prescribed
    // analytic mode is structural (O((h/R)^2) consistency error). If
    // any identity is violated at the percent level, there's a real
    // numerical bug in the basis/gradient/hessian computation that
    // would propagate into the membrane assembly.
    constexpr double R = 0.10, L = 2.00;
    constexpr int n_around = 48, n_along = 64;
    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    const Eigen::Index n_v = mesh.V.rows();

    // Build per-vertex adjacency once.
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n_v));
    for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            const int u = mesh.F(f, e), v = mesh.F(f, (e + 1) % 3);
            adj[static_cast<std::size_t>(u)].push_back(v);
            adj[static_cast<std::size_t>(v)].push_back(u);
        }
    }

    auto probe_anchor = [&](Eigen::Index anchor, const char* tag) {
    std::vector<int> nbr_ids;
    std::vector<int> ring_of(static_cast<std::size_t>(n_v), -1);
    ring_of[static_cast<std::size_t>(anchor)] = 0;
    nbr_ids.push_back(static_cast<int>(anchor));
    for (int ring = 1; ring <= 3; ++ring) {
        std::vector<int> next;
        for (int v : nbr_ids) {
            if (ring_of[static_cast<std::size_t>(v)] != ring - 1) continue;
            for (int w : adj[static_cast<std::size_t>(v)]) {
                if (ring_of[static_cast<std::size_t>(w)] == -1) {
                    ring_of[static_cast<std::size_t>(w)] = ring;
                    next.push_back(w);
                }
            }
        }
        nbr_ids.insert(nbr_ids.end(), next.begin(), next.end());
    }
    // Compute h_a the way the assembler does: AVERAGE of all one-ring
    // edge lengths through the anchor (vertex_one_ring_h in lme.cpp).
    // For our anisotropic cylinder: 2 circ edges + 2 axial edges + 2
    // diagonals — average ~26mm at this resolution, NOT just the
    // 13mm circumferential chord.
    double sum_edge = 0; int cnt_edge = 0;
    for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            if (mesh.F(f, e) != anchor) continue;
            const Eigen::Vector3d d =
                mesh.V.row(mesh.F(f, (e + 1) % 3)).transpose() -
                mesh.V.row(anchor).transpose();
            sum_edge += d.norm(); cnt_edge++;
        }
    }
    const double h_a = sum_edge / std::max(cnt_edge, 1);
    std::fprintf(stderr,
        "\n[lme_basis_consistency] === %s anchor %lld at "
        "(x=%.3f, y=%.3f, z=%.3f), %zu BFS-3-ring neighbours, "
        "h_a=%.3fmm ===\n",
        tag, (long long)anchor, mesh.V(anchor,0), mesh.V(anchor,1),
        mesh.V(anchor,2), nbr_ids.size(), h_a * 1000.0);
    REQUIRE(nbr_ids.size() >= 7);

    // β_wpca = γ_wpca / h_a^2  (use paper default γ_wpca = 1.8).
    Eigen::VectorXd beta_wpca = Eigen::VectorXd::Zero(n_v);
    for (int b : nbr_ids) beta_wpca(b) = 1.8 / (h_a * h_a);

    const auto patch = chladni::shell::lme::build_patch(
        static_cast<int>(anchor), mesh.V, nbr_ids, beta_wpca);

    // β_lme = γ_lme / h_a^2 on the chart (only over neighbours).
    Eigen::VectorXd beta_chart(static_cast<Eigen::Index>(nbr_ids.size()));
    for (Eigen::Index k = 0; k < beta_chart.size(); ++k)
        beta_chart(k) = 1.6 / (h_a * h_a);  // γ_lme = 1.6 default
    const double r_cut = 1.4 * h_a;

    auto check_point = [&](const Eigen::Vector2d& xi_g, const char* tag) {
        chladni::shell::LMEBasisGradHess gh;
        try {
            gh = chladni::shell::lme::evaluate_basis_grad_and_hess(
                patch.xi, beta_chart, xi_g, r_cut);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "  %-12s xi_g=(%.4f, %.4f)  --- LME BASIS FAILED: %s\n",
                tag, xi_g(0), xi_g(1), e.what());
            return;
        }
        const Eigen::Index nact = static_cast<Eigen::Index>(gh.indices.size());

        // Sum p_b, Σ ∂_α p_b, Σ ∂_α p_b · ξ_b^β.
        double sum_p = 0.0;
        Eigen::Vector2d sum_grad = Eigen::Vector2d::Zero();
        Eigen::Matrix2d sum_grad_xi = Eigen::Matrix2d::Zero();
        Eigen::Vector2d sum_p_xi = Eigen::Vector2d::Zero();
        for (Eigen::Index k = 0; k < nact; ++k) {
            const int lid = gh.indices[static_cast<std::size_t>(k)];
            const double p_k = gh.values[static_cast<std::size_t>(k)];
            const Eigen::VectorXd& dp_k = gh.gradients[static_cast<std::size_t>(k)];
            const Eigen::Vector2d xi_k = patch.xi.row(lid).head<2>().transpose();
            sum_p += p_k;
            sum_grad += dp_k;
            sum_grad_xi += dp_k * xi_k.transpose();
            sum_p_xi += p_k * xi_k;
        }

        std::fprintf(stderr,
            "  %-12s xi_g=(%.4f, %.4f)  n_active=%lld\n", tag,
            xi_g(0), xi_g(1), (long long)nact);
        std::fprintf(stderr,
            "    (a) Σ p_b              = %.6e   (should be 1.0)        |dev| = %.3e\n",
            sum_p, std::abs(sum_p - 1.0));
        std::fprintf(stderr,
            "    (b) Σ ∂_α p_b          = (%.3e, %.3e)  (should be 0)   |max| = %.3e\n",
            sum_grad(0), sum_grad(1), sum_grad.lpNorm<Eigen::Infinity>());
        std::fprintf(stderr,
            "    (c) Σ p_b·ξ_b          = (%.6f, %.6f)  (should be xi_g (%.4f, %.4f))   |dev| = %.3e\n",
            sum_p_xi(0), sum_p_xi(1), xi_g(0), xi_g(1),
            (sum_p_xi - xi_g).lpNorm<Eigen::Infinity>());
        std::fprintf(stderr,
            "    (d) Σ ∂_α p_b·ξ_b^β    = [%.6e, %.3e; %.3e, %.6e]\n"
            "                            (should be identity 2x2)   |dev| = %.3e\n",
            sum_grad_xi(0,0), sum_grad_xi(0,1),
            sum_grad_xi(1,0), sum_grad_xi(1,1),
            (sum_grad_xi - Eigen::Matrix2d::Identity()).lpNorm<Eigen::Infinity>());

        // (e) Reconstruct phi_alpha and compare to TRUE cylinder
        //     tangent at the LME-reconstructed Gauss point. This
        //     pins the chart-tilt error that the membrane assembly
        //     dot-products into ε_αβ.
        Eigen::Vector3d phi_1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d phi_2 = Eigen::Vector3d::Zero();
        Eigen::Vector3d P_recon = Eigen::Vector3d::Zero();
        for (Eigen::Index k = 0; k < nact; ++k) {
            const int lid = gh.indices[static_cast<std::size_t>(k)];
            const int gid = nbr_ids[static_cast<std::size_t>(lid)];
            const double p_k = gh.values[static_cast<std::size_t>(k)];
            const Eigen::VectorXd& dp_k = gh.gradients[static_cast<std::size_t>(k)];
            const Eigen::Vector3d Pb = mesh.V.row(gid).transpose();
            P_recon += p_k * Pb;
            phi_1 += dp_k(0) * Pb;
            phi_2 += dp_k(1) * Pb;
        }
        // True cylinder tangents at the LME-reconstructed Gauss point.
        const double th = std::atan2(P_recon(1), P_recon(0));
        const Eigen::Vector3d e_theta(-std::sin(th),  std::cos(th), 0.0);
        const Eigen::Vector3d e_z    ( 0.0, 0.0, 1.0);
        const Eigen::Vector3d e_r    ( std::cos(th),  std::sin(th), 0.0);
        // Decompose phi_1, phi_2 into (e_theta, e_z, e_r) components.
        const double phi1_theta = phi_1.dot(e_theta);
        const double phi1_z     = phi_1.dot(e_z);
        const double phi1_r     = phi_1.dot(e_r);    // ← normal/radial leak
        const double phi2_theta = phi_2.dot(e_theta);
        const double phi2_z     = phi_2.dot(e_z);
        const double phi2_r     = phi_2.dot(e_r);    // ← normal/radial leak
        std::fprintf(stderr,
            "    (e) phi_1 in (θ, z, r) = (%+.4f, %+.4f, %+.4f)  "
            "|phi_1|=%.4f\n",
            phi1_theta, phi1_z, phi1_r, phi_1.norm());
        std::fprintf(stderr,
            "        phi_2 in (θ, z, r) = (%+.4f, %+.4f, %+.4f)  "
            "|phi_2|=%.4f\n",
            phi2_theta, phi2_z, phi2_r, phi_2.norm());
        std::fprintf(stderr,
            "        RADIAL LEAK |phi_α·e_r|/|phi_α|: phi_1=%.3e   phi_2=%.3e\n",
            std::abs(phi1_r)/phi_1.norm(),
            std::abs(phi2_r)/phi_2.norm());
    };

    // Probe at representative INTERIOR Gauss-point chart coords.
    // (xi=(0,0) is the anchor node — gradient is undefined there per
    // LME's kink at conv-hull corners.) Triangle centroids inside
    // the chart's interior land at fractions of h_a away from any
    // node — these are what the assembler actually integrates over.
    check_point(Eigen::Vector2d(0.25*h_a, 0.10*h_a), "centroid-ish 1");
    check_point(Eigen::Vector2d(0.40*h_a, 0.30*h_a), "centroid-ish 2");
    check_point(Eigen::Vector2d(0.10*h_a, 0.40*h_a), "centroid-ish 3");
    check_point(Eigen::Vector2d(0.80*h_a, 0.60*h_a), "near patch edge");
    };  // probe_anchor

    // Two probes: interior anchor at z=L/2, and a free-boundary anchor at z=0.
    probe_anchor((n_along / 2) * n_around + 12, "INTERIOR (z=L/2)");
    probe_anchor(0                * n_around + 12, "BOUNDARY (z=0)");

    SUCCEED("LME basis consistency residuals printed to stderr");
}

TEST_CASE("cylinder PRESCRIBED inextensional n=2 ovalling — direct K_mem "
          "probe (Loop vs LME)",
          "[.diag][shell][cylinder][lme_prescribed_mem]")
{
    // Pin where LME generates spurious membrane: prescribe the
    // analytic INEXTENSIONAL n=2 ovalling displacement field at every
    // node and evaluate u^T K_mem u directly. The continuum mode is
    // u_r=cos(2θ), u_θ=-sin(2θ)/2, u_z=0; by elementary derivation it
    // satisfies ε_θθ=ε_zz=ε_θz=0 exactly (Love). On a discrete mesh,
    // both methods will give nonzero K_mem energy from chord/curvature
    // approximation, but the magnitudes characterise how each method
    // handles inextensional deformations:
    //   - small E_mem on Loop  -> Loop's subdivision basis recovers
    //                             the inextensional family
    //   - huge E_mem on LME    -> LME's chart-based gradient generates
    //                             spurious membrane strain on the
    //                             EXACT continuum mode shape (not just
    //                             on its own eigenmode); confirms the
    //                             bug is in the membrane B-matrix path
    //                             on cylindrical geometry, NOT in the
    //                             eigenmode-shape approximation.
    constexpr double R = 0.10, L = 2.00, h_th = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h_th);
    chladni::shell::ShellMaterial sm_mem = sm; sm_mem.k_B = 0.0;

    constexpr int n_around = 48, n_along = 64;
    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_v   = mesh.V.rows();
    const Eigen::Index n_dof = 3 * n_v;

    // Analytic ovalling displacement (Cartesian) at a point (x,y) on the
    // cylinder — reused for real vertices and (below) ghost nodes.
    auto oval = [](double x, double y) -> Eigen::Vector3d {
        const double th = std::atan2(y, x);
        const double cth = std::cos(th), sth = std::sin(th);
        const double ur = std::cos(2.0 * th);
        const double ut = -0.5 * std::sin(2.0 * th);
        return {ur * cth - ut * sth, ur * sth + ut * cth, 0.0};
    };
    // Prescribe the analytic ovalling at each vertex, in Cartesian.
    Eigen::VectorXd u_an = Eigen::VectorXd::Zero(n_dof);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        u_an.segment<3>(3 * v) = oval(mesh.V(v, 0), mesh.V(v, 1));
    }

    std::fprintf(stderr,
        "\n[lme_prescribed_mem] === Cylinder %dx%d, R=%.3f L=%.3f h=%.3e ===\n"
        "  Prescribed: analytic inextensional n=2 ovalling at each node.\n"
        "  Continuum membrane strain = 0 exactly. Expected E_mem -> 0.\n\n",
        n_around, n_along, R, L, h_th);

    auto report = [&](const char* tag, auto&& a_factory) {
        auto a_mem = a_factory();
        const auto K_mem = a_mem.assemble_K(mesh.V, mesh.F, sm_mem);
        // Slice u to match K_mem's row count (LME with ghost-on
        // returns 3(N+G); prescribed u covers only the real nodes).
        const Eigen::Index nrK = K_mem.rows();
        Eigen::VectorXd u_use = Eigen::VectorXd::Zero(nrK);
        u_use.head(std::min(nrK, n_dof)) =
            u_an.head(std::min(nrK, n_dof));
        const double E_mem = u_use.dot(K_mem * u_use);
        // Normalise by the mass-matrix norm so amplitudes are
        // comparable across methods (which mass-orthonormalise modes).
        auto a_M = a_factory();
        const double rho_h = mat.density * h_th;
        const auto M = a_M.assemble_M(mesh.V, mesh.F, rho_h);
        const double M_norm = u_use.dot(M * u_use);
        std::fprintf(stderr,
            "  %-15s  E_mem = u^T K_mem u = %10.4e   "
            "u^T M u = %.4e   E_mem / (u^T M u) = %.4e\n",
            tag, E_mem, M_norm, E_mem / std::max(M_norm, 1e-30));
    };
    // Support-width sweep via the LIVE knob γ (lower γ → wider, flatter
    // basis). r_cut_mult_curved is retired/inert (value-based cutoff),
    // so the old "rcut=1.4/2.0/3.0/4.0" rows were all identical — they
    // are replaced with a real γ sweep here.
    report("Loop",            []() { return chladni::shell::LoopAssembler{}; });
    report("LME(gamma=1.6,gf)", []() {
        chladni::shell::LMEAssembler::Params p;
        p.use_ghost_nodes = false; p.gamma = 1.6;
        return chladni::shell::LMEAssembler{p};
    });
    report("LME(gamma=1.0,gf)", []() {
        chladni::shell::LMEAssembler::Params p;
        p.use_ghost_nodes = false; p.gamma = 1.0;
        return chladni::shell::LMEAssembler{p};
    });
    report("LME(gamma=0.8,gf)", []() {
        chladni::shell::LMEAssembler::Params p;
        p.use_ghost_nodes = false; p.gamma = 0.8;
        return chladni::shell::LMEAssembler{p};
    });
    // 2nd-order SME (ghost-on; converges since the 2026-05-30 anisotropic
    // -gap fix). Paper §4.1.2 motivates SME as the remedy to LME's
    // inextensional locking. SME needs ghosts on the cylinder, so the
    // generic zero-pad-ghosts slicing in `report` would inject a spurious
    // boundary discontinuity. Instead prescribe the analytic ovalling at
    // the GHOST positions too (they sit on the cylinder continuation) so
    // u is the genuine inextensional shape over the full 3*(n_v+G) space
    // — a CLEAN locking measurement comparable to the Loop/LME rows.
    {
        const auto bdry = chladni::shell::lme::collect_boundary_edges(mesh.F);
        const Eigen::MatrixXd ghosts =
            chladni::shell::lme::build_ghost_positions(
                mesh.V, mesh.F, bdry);
        const Eigen::Index G = ghosts.rows();
        const Eigen::Index n_ext = 3 * (n_v + G);
        Eigen::VectorXd u_ext = Eigen::VectorXd::Zero(n_ext);
        u_ext.head(n_dof) = u_an;
        for (Eigen::Index g = 0; g < G; ++g) {
            u_ext.segment<3>(n_dof + 3 * g) =
                oval(ghosts(g, 0), ghosts(g, 1));
        }
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;   // ghost-on default
        chladni::shell::LMEAssembler a_mem{p};
        const auto K_mem = a_mem.assemble_K(mesh.V, mesh.F, sm_mem);
        REQUIRE(K_mem.rows() == n_ext);
        chladni::shell::LMEAssembler a_M{p};
        const auto M = a_M.assemble_M(mesh.V, mesh.F, mat.density * h_th);
        const double E_mem  = u_ext.dot(K_mem * u_ext);
        const double M_norm = u_ext.dot(M * u_ext);
        std::fprintf(stderr,
            "  %-15s  E_mem = u^T K_mem u = %10.4e   "
            "u^T M u = %.4e   E_mem / (u^T M u) = %.4e\n",
            "SME(gn,ghost-u)", E_mem, M_norm,
            E_mem / std::max(M_norm, 1e-30));
    }
    SUCCEED("Prescribed-inextensional E_mem printed to stderr");
}

TEST_CASE("cylinder PRESCRIBED inextensional ovalling vs mesh ASPECT — "
          "SME spurious-membrane scaling",
          "[.diag][shell][cylinder][sme_mem_aspect]")
{
    // Locking discriminator for the +45.6 % aspect-2.39 SME number
    // (2026-06-06). The [sme_cls]/[sme_beta] probes ACQUITTED the gap
    // classification (zero chart-shape kind flips), the drop net (max
    // 0.2 % of any Gauss point's PoU weight), and the rim slack
    // magnitudes (α- and β-insensitive). The remaining explanation is
    // the in-code one: genuine anisotropic membrane consistency error.
    // This probe pins it with the eigensolver removed: prescribe the
    // EXACT analytic inextensional n=2 ovalling (continuum E_mem = 0)
    // on the aspect ladder and compare the normalised spurious
    // membrane energy E_mem/(uᵀMu) of SME vs Loop. If SME's value
    // grows strongly with aspect while Loop's stays flat, the +45.6 %
    // is a genuine SME-on-anisotropic-grid consistency limit (the
    // paper's own §3.3 "accuracy wiggles on nonuniform grids" caveat),
    // NOT a classification artifact.
    constexpr double R = 0.10, L = 2.00, h_th = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h_th);
    chladni::shell::ShellMaterial sm_mem = sm; sm_mem.k_B = 0.0;

    auto oval = [](double x, double y) -> Eigen::Vector3d {
        const double th = std::atan2(y, x);
        const double cth = std::cos(th), sth = std::sin(th);
        const double ur = std::cos(2.0 * th);
        const double ut = -0.5 * std::sin(2.0 * th);
        return {ur * cth - ut * sth, ur * sth + ut * cth, 0.0};
    };

    constexpr int n_around = 24;
    const double circ_edge = (2.0 * std::numbers::pi_v<double> * R) / n_around;
    for (int n_along : {32, 48, 64, 73}) {
        const auto mesh =
            chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        const Eigen::Index n_v   = mesh.V.rows();
        const Eigen::Index n_dof = 3 * n_v;
        Eigen::VectorXd u_an = Eigen::VectorXd::Zero(n_dof);
        for (Eigen::Index v = 0; v < n_v; ++v) {
            u_an.segment<3>(3 * v) = oval(mesh.V(v, 0), mesh.V(v, 1));
        }
        const double aspect = (L / n_along) / circ_edge;

        // Loop reference.
        double loop_ratio = -1.0;
        {
            chladni::shell::LoopAssembler a_mem, a_M;
            const auto K_mem = a_mem.assemble_K(mesh.V, mesh.F, sm_mem);
            const auto M = a_M.assemble_M(mesh.V, mesh.F,
                                          mat.density * h_th);
            loop_ratio = u_an.dot(K_mem * u_an)
                       / std::max(u_an.dot(M * u_an), 1e-30);
        }
        // SME, ghost-on, analytic ovalling prescribed at ghosts too
        // (they sit on the cylinder continuation) — same protocol as
        // [lme_prescribed_mem]'s SME row.
        double sme_ratio = -1.0;
        {
            const auto bdry =
                chladni::shell::lme::collect_boundary_edges(mesh.F);
            const Eigen::MatrixXd ghosts =
                chladni::shell::lme::build_ghost_positions(
                    mesh.V, mesh.F, bdry);
            const Eigen::Index G = ghosts.rows();
            Eigen::VectorXd u_ext = Eigen::VectorXd::Zero(3 * (n_v + G));
            u_ext.head(n_dof) = u_an;
            for (Eigen::Index g = 0; g < G; ++g) {
                u_ext.segment<3>(n_dof + 3 * g) =
                    oval(ghosts(g, 0), ghosts(g, 1));
            }
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            chladni::shell::LMEAssembler a_mem{p}, a_M{p};
            const auto K_mem = a_mem.assemble_K(mesh.V, mesh.F, sm_mem);
            REQUIRE(K_mem.rows() == 3 * (n_v + G));
            const auto M = a_M.assemble_M(mesh.V, mesh.F,
                                          mat.density * h_th);
            sme_ratio = u_ext.dot(K_mem * u_ext)
                      / std::max(u_ext.dot(M * u_ext), 1e-30);
        }
        std::fprintf(stderr,
            "[sme_mem_aspect] n_along=%-3d aspect=%4.2f  "
            "E_mem/(uMu): Loop %.4e   SME %.4e   SME/Loop %.1f\n",
            n_along, aspect, loop_ratio, sme_ratio,
            sme_ratio / std::max(loop_ratio, 1e-30));
    }
    // MODAL membrane fraction via Hellmann-Feynman: for an eigenpair
    // (ω², u) of (K_bend + k_L·K_mem/k_L, M), the logarithmic
    // sensitivity d(ln ω²)/d(ln k_L) equals the fraction of modal
    // strain energy carried by the membrane operator. A true
    // inextensional ovalling mode has fraction ≈ 0; membrane LOCKING
    // (the discrete space failing to represent a low-membrane ovalling
    // shape) shows up as a large fraction. Works through the public
    // API (no ghost-DOF mode slicing needed): two eigensolves at k_L
    // and 0.9·k_L, central-difference the log frequencies.
    std::fprintf(stderr, "[sme_mem_aspect] modal membrane fraction "
                 "d(ln w^2)/d(ln k_L) (HF):\n");
    for (int n_along : {32, 73}) {
        const auto mesh =
            chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        const double aspect = (L / n_along) / circ_edge;
        auto omega_n2 = [&](double kL_scale, auto&& a_factory) {
            chladni::shell::ShellMaterial sm_s = sm;
            sm_s.k_L *= kL_scale;
            auto a = a_factory();
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm_s, h_th, 2, a);
            return 0.5 * (m.omegas(0) + m.omegas(1));
        };
        auto frac = [&](auto&& a_factory) {
            const double w1 = omega_n2(1.0, a_factory);
            const double w9 = omega_n2(0.9, a_factory);
            return std::pair<double, double>{
                w1,
                2.0 * std::log(w1 / w9) / std::log(1.0 / 0.9)};
        };
        const auto [w_loop, f_loop] =
            frac([]() { return chladni::shell::LoopAssembler{}; });
        const auto [w_sme, f_sme] = frac([]() {
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            return chladni::shell::LMEAssembler{p};
        });
        std::fprintf(stderr,
            "[sme_mem_aspect]   n_along=%-3d aspect=%4.2f  "
            "Loop w=%7.1f mem_frac=%.3f   SME w=%7.1f mem_frac=%.3f\n",
            n_along, aspect, w_loop, f_loop, w_sme, f_sme);
    }
    SUCCEED("Prescribed-inextensional aspect scaling printed to stderr");
}

TEST_CASE("cylinder LME geometric SCALE invariance (do edge lengths "
          "enter dimensionally?)",
          "[.diag][shell][modes][cylinder][lme_scale]")
{
    // Probe whether LME's K and M scale correctly with the absolute
    // edge length. Continuum thin-shell theory with fixed thickness h:
    // under uniform in-plane scaling V -> s*V, K is invariant (membrane
    // and bending stiffnesses are both edge-length-invariant), M scales
    // as s^2, so omega = sqrt(K/M) scales as 1/s. If LME has a hidden
    // absolute-length scale anywhere (a Newton-convergence tolerance
    // measured in raw meters, an r_cut comparison against an unscaled
    // constant, a gamma parameter that should be dimensionless but isn't,
    // a numerical-rank threshold set in absolute units), scaling will
    // shift the LME/Loop ratio while Loop's ratio (the continuum
    // reference) stays at 1 to numerical precision.
    //
    // Three scales: s = 0.5, 1.0, 2.0. Same mesh topology (48x64) =>
    // identical sparsity pattern, identical edge-length-ratios; the
    // ONLY thing that changes is the absolute length of every edge.
    constexpr double R0 = 0.10, L0 = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    constexpr int n_around = 48, n_along = 64;
    const std::array<double, 3> scales{0.5, 1.0, 2.0};

    std::fprintf(stderr,
        "\n[lme_scale] === LME vs Loop under uniform geometric scaling ===\n"
        "  same mesh topology (%dx%d), thickness h=%.3e fixed.\n"
        "  Expectation: omega ~ 1/scale; LME/Loop ratio constant if LME\n"
        "  is dimensionally clean. Smallest edge prints in mm.\n\n",
        n_around, n_along, h);

    for (double s : scales) {
        const double R = R0 * s, L = L0 * s;
        const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        // Smallest edge for scale context.
        double e_min = 1e30;
        for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
            for (int e = 0; e < 3; ++e) {
                const Eigen::Vector3d d =
                    mesh.V.row(mesh.F(f, e)) -
                    mesh.V.row(mesh.F(f, (e + 1) % 3));
                e_min = std::min(e_min, d.norm());
            }
        }

        // Split K into membrane (k_L) and bending (k_B) parts so we
        // can evaluate u^T K_mem u and u^T K_bend u separately for each
        // method's lowest n=2 ovalling mode. For a near-inextensional
        // mode (which n=2 ovalling at L/R=20 should be) the expected
        // ratio is E_mem/E_bend ~ (h/R)^2 — see Love. Anomalously large
        // membrane on LME = membrane locking.
        chladni::shell::ShellMaterial sm_mem = sm; sm_mem.k_B = 0.0;
        chladni::shell::ShellMaterial sm_bnd = sm; sm_bnd.k_L = 0.0;

        auto split = [&](const char* tag, auto&& a_factory) {
            auto a_full = a_factory();
            // Mode shape (the eigenmode of the full operator).
            const auto modes = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 1, a_full);
            const double omega = modes.omegas(0);
            const Eigen::VectorXd u = modes.shapes.col(0);

            // Separately assemble the membrane-only and bending-only K
            // with the SAME assembler/parameters.
            auto a_mem = a_factory();
            auto a_bnd = a_factory();
            const auto K_mem = a_mem.assemble_K(mesh.V, mesh.F, sm_mem);
            const auto K_bnd = a_bnd.assemble_K(mesh.V, mesh.F, sm_bnd);
            const double E_mem = u.dot(K_mem * u);
            const double E_bnd = u.dot(K_bnd * u);

            const double analytic_inext_ratio = (h / R) * (h / R);
            std::fprintf(stderr,
                "  %-10s ω=%8.2f   uKmem u=%9.3e   uKbnd u=%9.3e   "
                "E_mem/E_bnd=%.3e   (expected ~(h/R)^2=%.3e)\n",
                tag, omega, E_mem, E_bnd, E_mem / E_bnd,
                analytic_inext_ratio);
        };
        std::fprintf(stderr,
            "[lme_scale] s=%.2f  R=%.4f  edge=%.3fmm  (h/R=%.4f, "
            "(h/R)^2=%.3e)\n", s, R, e_min * 1.0e3, h / R, (h/R)*(h/R));
        split("Loop",         []() { return chladni::shell::LoopAssembler{}; });
        // Shipped-default LME (curved + value-based truncation, ghost-off
        // here so the K_mem/K_bnd energy split matches the 3*n_v mode
        // shape). Bending is now resolved correctly by the value-based
        // cutoff (Params::tol_lme); r_cut_mult_curved is inert. This line
        // ISOLATES the residual membrane locking once bending is right:
        // E_mem/E_bnd stays O(1) at thin scales (vs (h/R)^2 expected),
        // and LME/Loop ω drifts with scale — the inextensional-membrane
        // locking, NOT fixed by the bending correction.
        split("LME", []() {
            chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
            return chladni::shell::LMEAssembler{p};
        });
        // NOTE (2026-05-29): selective REDUCED MEMBRANE INTEGRATION
        // (centroid 1-point rule on K_mem) was implemented and tested here
        // and DOES NOT cure this locking — it changed the spurious membrane
        // energy by only ~2-4%. The 1-point rule is exact for constant/
        // linear fields, so "reduced ≈ full" means the spurious LME
        // membrane strain is a near-CONSTANT field present even at the
        // centroid, not the high-order parasitic strain reduced integration
        // removes. The locking is an intrinsic 1st-order B_mem consistency
        // defect (paper §4.1.2), curable only by 2nd-order SME. The flag was
        // reverted (trap-knob discipline). See chladni-lme-cylinder-locking.
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr,
        "\n[lme_scale] Loop scale ratio s=2.0 / s=1.0 omega: should be 0.5.\n"
        "[lme_scale] LME  scale ratio s=2.0 / s=1.0 omega: should be 0.5 if\n"
        "                  LME is dimensionally clean.\n");
    SUCCEED("LME geometric-scale invariance check printed to stderr");
}

TEST_CASE("cylinder SME convergence vs mesh ASPECT (isotropic-gap "
          "feasibility hypothesis)",
          "[.diag][.slow][shell][modes][cylinder][sme_aspect]")
{
    // REGRESSION GUARD: SME's in-chart Newton USED to diverge on the
    // anisotropic free-free cylinder at the paper's default α=2, β=1. Root
    // cause (2026-06-03 audit): the nodal gaps used a single SCALAR spacing
    // h_a = MEAN one-ring edge for every direction, so the near-free-edge
    // normal slack β·h_mean² under-covered the much larger AXIAL spacing on
    // the axially-graded mesh → the §3.2.2 feasibility condition along the
    // sparse (axial) direction failed → "every Shepard-active patch
    // diverged". The failure was β-sensitive, NOT α-sensitive, pinning it to
    // the boundary normal term.
    //
    // FIXED faithfully (RMA13 §3.2.1–§3.2.2) by using the DIRECTIONAL nodal
    // spacing h_a(u) = max_b |(x_b−x_a)·u| (the paper's max-adjacent rule) in
    // each gap direction — interior eigen-directions and boundary n/t alike —
    // instead of the trace-normalised covariance × mean-edge. The cylinder
    // now CONVERGES at every aspect at the paper's α=2, β=1, with NO mesh
    // change, and the milder aspects also got more accurate (aspect 1.01:
    // +10.7% → +7.7%; 1.19: +11.7% → +8.4%). The residual over-stiffness at
    // high aspect (≈+46% at aspect 2.39) is the inextensional MEMBRANE
    // LOCKING — a consistency limit the gap does not touch, and the free-free
    // cylinder is the one case the papers never test (they use diaphragm
    // ends). CONFIRMED MODALLY 2026-06-06 ([sme_mem_aspect]): the n=2
    // mode's Hellmann-Feynman membrane energy fraction reads 0.444 at
    // aspect 2.39 (Loop: 0.039) and 0.019 at aspect 1.05 (cleaner than
    // Loop's 0.046); classification/drop-net/rim-slack alternatives all
    // acquitted by direct measurement ([sme_cls], [sme_beta]). Also
    // relies on the faithful per-patch quadrature ownership filter
    // (Millán 2011 §4.1.1).
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const double rayleigh_n2 =
        chladni::analytical::
            free_free_cylindrical_shell_inextensional_angular_frequencies(
                geom, mat, 1)[0];

    constexpr int n_around = 24;
    const double circ_edge = (2.0 * std::numbers::pi_v<double> * R) / n_around;
    // Axial edge = L / n_along; isotropic when n_along ~= L / circ_edge.
    const int n_along_iso =
        static_cast<int>(std::lround(L / circ_edge));   // ~73 here
    const std::array<int, 4> n_along_sweep{
        32, 48, 64, n_along_iso};

    std::fprintf(stderr,
        "\n[sme_aspect] === SME convergence vs cylinder mesh aspect ===\n"
        "  R=%.2f L=%.2f h=%.0e, n_around=%d (circ edge %.2f mm).\n"
        "  Isotropic at n_along~=%d. Rayleigh n=2 = %.1f rad/s.\n"
        "  aspect = axial_edge / circ_edge (1.0 = isotropic).\n\n",
        R, L, h, n_around, circ_edge * 1e3, n_along_iso, rayleigh_n2);

    for (int n_along : n_along_sweep) {
        const auto mesh =
            chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        const double axial_edge = L / n_along;
        const double aspect      = axial_edge / circ_edge;

        auto run = [&](const char* tag, auto&& a_factory) -> std::string {
            try {
                auto a = a_factory();
                const auto m = chladni::shell::compute_shell_modes(
                    mesh.V, mesh.F, mat, sm, h, 2, a);
                const double w = 0.5 * (m.omegas(0) + m.omegas(1));
                char buf[64];
                std::snprintf(buf, sizeof buf, "%8.1f (%+6.1f%%)",
                    w, 100.0 * (w - rayleigh_n2) / rayleigh_n2);
                return buf;
            } catch (const std::exception&) {
                return "  DIVERGED";
            }
        };

        const std::string loop = run("Loop",
            []() { return chladni::shell::LoopAssembler{}; });
        // LME-1st column (matrix §3): does the 1st-order membrane
        // locking ease with aspect, or stay catastrophic at every
        // level? (The matrix previously had this only at the coarsest
        // aspect.)
        const std::string lme = run("LME-1st",
            []() { return chladni::shell::LMEAssembler{
                chladni::shell::LMEAssembler::Params{}}; });
        const std::string sme = run("SME", []() {
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            return chladni::shell::LMEAssembler{p};
        });
        // Chart A/B re-run column (2026-06-07): the round-3 intrinsic
        // verdict (loses the ladder at finer levels, non-monotonic)
        // PREDATES interior ownership (6214ed0) — with the fringe
        // evaluation garbage gone the comparison must be re-read, per
        // the faithful-first "single-variant failures don't close
        // combinations" rule.
        const std::string sme_intr = run("SME intr3", []() {
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme  = true;
            p.sme_chart_radius_mult = 3.0;
            return chladni::shell::LMEAssembler{p};
        });
        std::fprintf(stderr,
            "  n_along=%-4d aspect=%4.2f  V=%-5lld  Loop %s   "
            "LME-1st %s   SME %s   SME-intr3 %s\n",
            n_along, aspect,
            static_cast<long long>(mesh.V.rows()),
            loop.c_str(), lme.c_str(), sme.c_str(), sme_intr.c_str());
    }
    std::fprintf(stderr,
        "\n[sme_aspect] POST-FIX: SME should CONVERGE at every aspect (the\n"
        "  anisotropic-gap fix). Residual over-stiffness vs Loop is the\n"
        "  unfixed inextensional membrane locking, not the gap feasibility.\n");
    // Guard the fix: SME must no longer diverge on the most anisotropic
    // mesh (aspect ~2.4), where it previously threw.
    {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, 32);
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        chladni::shell::LMEAssembler a{p};
        REQUIRE_NOTHROW(chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, 2, a));
    }
    SUCCEED("SME mesh-aspect convergence sweep printed to stderr");
}

TEST_CASE("cylinder SME gap-classification chart-sensitivity stats",
          "[.diag][shell][cylinder][sme_cls]")
{
    // Shape-robust-classification groundwork (2026-06-06): the chart
    // A/B rounds (scaffolding inventory C4, rounds 1-3) left the
    // hypothesis that SME's chart-LOCAL gap projection layer — the
    // kChartProjMin renormalise-or-fallback and the iso/aniso eigen
    // gate in build_per_patch_sme_gaps — responds DISCONTINUOUSLY to
    // chart shape, k-ring charts being the shape it was tuned on. This
    // probe anchors that hypothesis in counts instead of modal deltas:
    // it dumps the LME_DIAG-gated [sme-cls] classification tallies
    // (final-kind histogram, fallback counts, accepted-projection
    // extremes, per-vertex kind flips across charts) on the cylinder
    // aspect ladder under the legacy k-ring-3 chart vs the
    // intrinsic-geodesic extractor. Assembly-only — the gap builder
    // runs at the top of the curved assemble_K; no eigensolve needed.
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    // The tallies are LME_DIAG-gated; bracket the env var.
    const char* prev = std::getenv("LME_DIAG");
    setenv("LME_DIAG", "1", 1);

    constexpr int n_around = 24;
    const double circ_edge = (2.0 * std::numbers::pi_v<double> * R) / n_around;
    for (int n_along : {32, 73}) {
        const auto mesh =
            chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        for (double mult : {0.0, 3.0}) {
            std::fprintf(stderr,
                "\n[sme_cls] === n_along=%d (aspect %.2f) %s ===\n",
                n_along, (L / n_along) / circ_edge,
                mult == 0.0 ? "k-ring-3" : "intrinsic mult=3");
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme  = true;
            p.sme_chart_radius_mult = mult;
            chladni::shell::LMEAssembler a{p};
            (void)a.assemble_K(mesh.V, mesh.F, sm);
        }
    }
    if (prev) setenv("LME_DIAG", prev, 1);
    else      unsetenv("LME_DIAG");
    SUCCEED("SME gap-classification stats printed to stderr");
}

TEST_CASE("SME cylinder curved-K: ZERO in-chart Newton drops "
          "(B1 net dormant)",
          "[.slow][shell][sme][cylinder][validation][sme_no_drops]")
{
    // Faithful-first gate (2026-06-07). Root cause of the universal
    // rim-anchor SME Newton failures ([sme_rim_dissect]): at Gauss
    // points near a chart's coverage fringe the node cloud is
    // one-sided and the SME moment-with-gap program is GENUINELY
    // infeasible (certified separating hyperplane; gap enlargement
    // makes it worse; one extra ring of surround cures it). The paper
    // never meets the case — its patch neighbourhoods (basis-tol
    // sized, ~5.4h) dwarf its PU quadrature reach (~1.9h). Our fix
    // restores that invariant on the quadrature side: INTERIOR
    // ownership (triangle vertices + their 1-rings ⊆ chart) for the
    // SME path. This gate pins the consequence: the B1
    // drop-and-renormalise net — the one remaining load-bearing
    // invention — must be DORMANT for SME on the cylinder at
    // production parameters, exactly as it already is for LME.
    // Before the fix this read n_newton=17184 (aspect 2.39) /
    // 7056 (isotropic).
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    constexpr int n_around = 24;
    for (int n_along : {32, 73}) {
        const auto mesh =
            chladni::mesh::generate_cylinder(R, L, n_around, n_along);
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        chladni::shell::LMEAssembler a{p};
        (void)a.assemble_K(mesh.V, mesh.F, sm);
        const auto& ds = a.last_drop_stats();
        INFO("n_along=" << n_along
             << "  newton drops n=" << ds.n_newton
             << " (w=" << ds.w_newton
             << ")  ownership n=" << ds.n_own << " (w=" << ds.w_own
             << ")  escal rescues=" << ds.n_escal_ok
             << "  per-gauss max=" << ds.w_gauss_max);
        REQUIRE(ds.n_newton == 0);
    }
}

TEST_CASE("cylinder SME rim-drop weight vs sme_beta (normal-slack "
          "under-coverage hypothesis)",
          "[.diag][.slow][shell][modes][cylinder][sme_beta]")
{
    // Follow-up to [sme_cls] (2026-06-06): at aspect 2.39 under the
    // k-ring chart, the curved assemble_K silently sheds PoU weight
    // w=3.34 to in-chart Newton FAILURES + w=2.82 to ownership
    // exclusions, concentrated one row in from the rims — and the α
    // escalation rescues almost none of it (48 of 17184), consistent
    // with the 2026-06-03 audit's "β-sensitive, NOT α-sensitive"
    // failure signature: the §3.2.2 rim recipe gives boundary-EDGE
    // nodes zero slack across the boundary, and α scales only the
    // tangential term. If the dropped weight and the n=2 error fall
    // monotonically with sme_beta (which enlarges the NORMAL slack of
    // the near-rim rows), the +45.6 % aspect-2.39 number is (partly)
    // a rim-feasibility artifact of the gap VALUES on a curved rim,
    // not membrane locking — actionable by a curvature-aware
    // (sagitta) boundary gap rather than a knob.
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const double rayleigh_n2 =
        chladni::analytical::
            free_free_cylindrical_shell_inextensional_angular_frequencies(
                geom, mat, 1)[0];

    const char* prev = std::getenv("LME_DIAG");
    setenv("LME_DIAG", "1", 1);

    constexpr int n_around = 24;
    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, 32);
    for (double beta : {1.0, 2.0, 4.0}) {
        std::fprintf(stderr, "\n[sme_beta] === aspect 2.39, sme_beta=%.1f"
                             " ===\n", beta);
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.sme_beta             = beta;
        try {
            chladni::shell::LMEAssembler a{p};
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 2, a);
            const double w = 0.5 * (m.omegas(0) + m.omegas(1));
            std::fprintf(stderr,
                "[sme_beta] beta=%.1f  n=2 omega=%.1f  rel_err=%+.1f%%\n",
                beta, w, 100.0 * (w - rayleigh_n2) / rayleigh_n2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[sme_beta] beta=%.1f  FAILED: %s\n",
                         beta, e.what());
        }
    }
    if (prev) setenv("LME_DIAG", prev, 1);
    else      unsetenv("LME_DIAG");
    SUCCEED("SME beta sweep printed to stderr");
}

TEST_CASE("cylinder SME over-stiffness vs gap magnitude (sme_alpha sweep)",
          "[.diag][.slow][shell][modes][cylinder][sme_alpha]")
{
    // SME converges on the cylinder now (anisotropic gaps) but is still
    // ~+59% over-stiff at best vs Loop's +5% — does it actually CURE the
    // inextensional locking or just lessen it? Paper (Rosolen-Millán-
    // Arroyo 2013): the aspect ratio of the SME shape functions grows
    // with the gap magnitude relative to h² (set by sme_alpha, default
    // 4). A WIDER, smoother basis should represent the inextensional
    // bending with less spurious membrane → less locking. Sweep
    // sme_alpha on the near-isotropic cylinder (where SME converges) and
    // watch the n=2 over-stiffness. omega is the physical (ghost-on,
    // vertex-re-evaluated) eigenfrequency, so it is a valid measure
    // (unlike the u^T K_mem u split, which is confounded by the ghost
    // dof count — see [lme_scale] which forces ghost-off for that).
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);
    const chladni::analytical::CylindricalShell geom{
        .radius = R, .length = L, .thickness = h};
    const double rayleigh_n2 =
        chladni::analytical::
            free_free_cylindrical_shell_inextensional_angular_frequencies(
                geom, mat, 1)[0];

    constexpr int n_around = 24, n_along = 76;   // ~isotropic (aspect ~1)
    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);

    std::fprintf(stderr,
        "\n[sme_alpha] === SME n=2 over-stiffness vs sme_alpha ===\n"
        "  isotropic cylinder %dx%d, Rayleigh n=2 = %.1f rad/s.\n"
        "  larger alpha = larger gaps = wider/smoother SME basis.\n\n",
        n_around, n_along, rayleigh_n2);

    for (double alpha : {2.0, 4.0, 8.0, 16.0, 32.0, 64.0}) {
        std::string out;
        try {
            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            p.sme_alpha            = alpha;
            chladni::shell::LMEAssembler a{p};
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 2, a);
            const double w = 0.5 * (m.omegas(0) + m.omegas(1));
            char buf[64];
            std::snprintf(buf, sizeof buf, "%8.1f (%+6.1f%%)",
                w, 100.0 * (w - rayleigh_n2) / rayleigh_n2);
            out = buf;
        } catch (const std::exception& e) {
            out = std::string("DIVERGED — ") + e.what();
        }
        std::fprintf(stderr, "  sme_alpha=%5.1f  ω=%s\n", alpha, out.c_str());
    }
    std::fprintf(stderr,
        "\n[sme_alpha] If over-stiffness drops toward Loop (~+5%%) as alpha\n"
        "  grows, the gap magnitude is the lever that cures SME locking.\n");
    SUCCEED("SME sme_alpha sweep printed to stderr");
}

TEST_CASE("cylinder LME mesh-aspect SENSITIVITY (Loop converges on these "
          "same meshes)",
          "[.diag][shell][modes][cylinder][lme_mesh_aspect]")
{
    // The [mesh_geom] test shows the cylinder mesh has edge-anisotropy
    // 2.35-2.59x across the 24x32 -> 64x96 refinement table (axial
    // edges ~2.5x the circumferential chord, because L/R=20 forces
    // n_along/n_around >> 1 for uniform-along-axis discretisation).
    // Loop converges cleanly on this anisotropic mesh — see
    // [loop_vs_lme_refine] — so the mesh is mathematically fine, not
    // a degenerate triangulation. A well-formulated FEM should not be
    // mesh-aspect-sensitive.
    //
    // This test rebuilds the LME refinement table at *isotropic*
    // aspect (n_along/n_around ~ L/(2πR) ~ 3.18 for this cylinder),
    // DOF-matched to the existing anisotropic levels so the comparison
    // is apples-to-apples on stiffness-matrix size. If LME tracks Loop
    // on isotropic mesh but sinks on anisotropic mesh, the bug is an
    // LME mesh-aspect sensitivity (likely in the chart-blended Shepard
    // PoU or a quadrature-vs-basis-anisotropy interaction). If LME
    // sinks on both, the deficit is in the bending/membrane operator
    // proper and aspect is a red herring.
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    struct Lv { int na; int nl; const char* tag; };
    // Pairs chosen so n_along/n_around ~ 3.18 (isotropic triangles
    // for L/R=20), DOF count matches the existing anisotropic table.
    const std::array<Lv, 4> isotropic{{
        {14,  45, "iso14x45  (anisotropic-twin: 24x32, 792 V)"},
        {22,  70, "iso22x70  (anisotropic-twin: 32x48, 1568 V)"},
        {31,  99, "iso31x99  (anisotropic-twin: 48x64, 3120 V)"},
        {44, 140, "iso44x140 (anisotropic-twin: 64x96, 6208 V)"}}};

    std::fprintf(stderr,
        "\n[lme_mesh_aspect] === LME vs Loop on ISOTROPIC meshes ===\n"
        "  (compare against anisotropic refinement table [loop_vs_lme_refine]:\n"
        "    24x32: Loop 428.9  LME 1390.5\n"
        "    32x48: Loop 416.9  LME  633.9\n"
        "    48x64: Loop 412.3  LME  414.4   <- LME crosses ref\n"
        "    64x96: Loop 411.2  LME  342.4   <- LME 17%% deficit)\n\n");

    for (auto lv : isotropic) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, lv.na, lv.nl);
        const Eigen::Index n_v = mesh.V.rows();

        // Quick edge-aspect printout to confirm "iso" is iso.
        double e_min = 1e30, e_max = 0;
        for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
            for (int e = 0; e < 3; ++e) {
                const Eigen::Vector3d d =
                    mesh.V.row(mesh.F(f, e)) -
                    mesh.V.row(mesh.F(f, (e + 1) % 3));
                const double L_ = d.norm();
                e_min = std::min(e_min, L_); e_max = std::max(e_max, L_);
            }
        }
        std::fprintf(stderr,
            "[lme_mesh_aspect] %s   V=%lld  edge max/min=%.2fx\n",
            lv.tag, (long long)n_v, e_max / e_min);

        const auto t0 = std::chrono::steady_clock::now();
        double w_lme = 0.0, w_loop = 0.0;
        {
            chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
            chladni::shell::LMEAssembler a{p};
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 1, a);
            w_lme = m.omegas(0);
        }
        const auto t_lme = std::chrono::steady_clock::now();
        {
            chladni::shell::LoopAssembler a;
            const auto m = chladni::shell::compute_shell_modes(
                mesh.V, mesh.F, mat, sm, h, 1, a);
            w_loop = m.omegas(0);
        }
        const auto t_loop = std::chrono::steady_clock::now();
        const double lme_ms = std::chrono::duration<double, std::milli>(
            t_lme - t0).count();
        const double loop_ms = std::chrono::duration<double, std::milli>(
            t_loop - t_lme).count();
        std::fprintf(stderr,
            "    Loop ω = %7.2f  LME ω = %7.2f   ratio LME/Loop = %.3f"
            "   (LME %.0f ms, Loop %.0f ms)\n\n",
            w_loop, w_lme, w_lme / w_loop, lme_ms, loop_ms);
    }
    SUCCEED("isotropic-mesh LME refinement printed to stderr");
}

TEST_CASE("cylinder mesh GEOMETRY sanity (R, edges, areas, angles)",
          "[.diag][shell][cylinder][mesh_geom]")
{
    // Sanity check that the meshes the LME bug-hunt is using are
    // geometrically clean: every vertex on radius R, no degenerate
    // triangles, edge anisotropy quantified. The cylinder mesh is a
    // tensor-product (theta x z) grid with one diagonal direction —
    // helical/chiral triangulation, see [[chladni-cylinder-chirality
    // -2026-05-14]]. This test prints stats so we can rule out a
    // geometric defect before chasing the LME operator further.
    constexpr double R = 0.10, L = 2.00;
    const std::array<std::pair<int,int>, 4> levels{{
        {24, 32}, {32, 48}, {48, 64}, {64, 96}}};
    for (auto [na, nl] : levels) {
        const auto mesh = chladni::mesh::generate_cylinder(R, L, na, nl);
        const Eigen::Index n_v = mesh.V.rows();
        const Eigen::Index n_f = mesh.F.rows();

        double rdev_max = 0, zmin = 1e30, zmax = -1e30;
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const double r = std::sqrt(mesh.V(v,0)*mesh.V(v,0)
                                     + mesh.V(v,1)*mesh.V(v,1));
            rdev_max = std::max(rdev_max, std::abs(r - R));
            zmin = std::min(zmin, mesh.V(v,2));
            zmax = std::max(zmax, mesh.V(v,2));
        }

        std::vector<double> edges; edges.reserve(static_cast<std::size_t>(3 * n_f));
        std::vector<double> areas; areas.reserve(static_cast<std::size_t>(n_f));
        std::vector<double> angs;  angs.reserve(static_cast<std::size_t>(3 * n_f));
        double Atot = 0;
        for (Eigen::Index f = 0; f < n_f; ++f) {
            const Eigen::Vector3d a = mesh.V.row(mesh.F(f,0));
            const Eigen::Vector3d b = mesh.V.row(mesh.F(f,1));
            const Eigen::Vector3d c = mesh.V.row(mesh.F(f,2));
            const Eigen::Vector3d e0 = b - a, e1 = c - b, e2 = a - c;
            edges.push_back(e0.norm());
            edges.push_back(e1.norm());
            edges.push_back(e2.norm());
            const double A = 0.5 * ((b - a).cross(c - a)).norm();
            areas.push_back(A); Atot += A;
            for (auto [u, v] : std::array<std::pair<Eigen::Vector3d, Eigen::Vector3d>, 3>{{
                {e0, Eigen::Vector3d(-e2)},
                {e1, Eigen::Vector3d(-e0)},
                {e2, Eigen::Vector3d(-e1)}}}) {
                const double c_ = std::clamp(
                    u.normalized().dot(v.normalized()), -1.0, 1.0);
                angs.push_back(std::acos(c_) * 180.0 / std::numbers::pi);
            }
        }
        std::sort(edges.begin(), edges.end());
        std::sort(areas.begin(), areas.end());
        std::sort(angs.begin(),  angs.end());

        const double e_circ  = 2.0 * R * std::sin(std::numbers::pi / na);
        const double e_axial = L / nl;
        const double e_diag  = std::sqrt(e_circ*e_circ + e_axial*e_axial);
        const double A_cyl   = 2.0 * std::numbers::pi * R * L;

        std::fprintf(stderr,
            "[mesh_geom] === %dx%d (V=%lld F=%lld) ===\n", na, nl,
            (long long)n_v, (long long)n_f);
        std::fprintf(stderr,
            "  radius:     max|r-R|=%.3e   (R=%.3f)   z in [%.3f, %.3f] (L=%.3f)\n",
            rdev_max, R, zmin, zmax, L);
        std::fprintf(stderr,
            "  edges:      min=%.5e  med=%.5e  max=%.5e   max/min=%.2fx\n",
            edges.front(), edges[edges.size()/2], edges.back(),
            edges.back() / edges.front());
        std::fprintf(stderr,
            "  expected:   circ-chord=%.5e   axial=%.5e   diag=%.5e\n",
            e_circ, e_axial, e_diag);
        std::fprintf(stderr,
            "  areas:      min=%.4e   max=%.4e   total=%.5e   continuum=%.5e (rel diff %.3e)\n",
            areas.front(), areas.back(), Atot, A_cyl,
            (A_cyl - Atot) / A_cyl);
        std::fprintf(stderr,
            "  angles deg: min=%.2f  10%%=%.2f  med=%.2f  90%%=%.2f  max=%.2f\n",
            angs.front(),
            angs[angs.size()/10], angs[angs.size()/2],
            angs[angs.size()*9/10], angs.back());
    }
    SUCCEED("cylinder mesh geometry stats printed to stderr");
}

TEST_CASE("cylinder n=2 — LME spurious-mode 2D Fourier decomposition (m, k)",
          "[.diag][shell][modes][cylinder][lme_spurious_decomp]")
{
    // Lead #2 from session 2026-05-28 handoff
    // [[chladni-session-handoff-2026-05-28]]: the previous
    // [cross_operator] test shows LME admits a soft n=2 shape that
    // Loop's K rates at 13x the LME-K cost. Cross-operator confirms
    // it is a shape/operator mismatch, not a mass bug, but does not
    // pin DOWN which direction in shape space LME mis-rates.
    //
    // The cylinder mesh is a tensor product (theta x z) grid: vertex
    // (j*n_around + i) sits at (theta_i = 2pi i/n_around,
    // z_j = L j/n_along). So a 2D Fourier decomposition of u_r,
    // u_theta, u_z (cylindrical coords) is exact and cheap. The true
    // free-free n=2 inextensional mode is u_r = cos(2theta + phi),
    // u_theta = -sin(2theta + phi)/2, u_z = 0, uniform in z (k=0).
    // Anything else in the LME lowest mode tells us where the
    // spurious shape sits.
    constexpr double R = 0.10, L = 2.00, h = 1.00e-3;
    const auto mat = steel();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, h);

    constexpr int n_around = 64;
    constexpr int n_along  = 96;
    constexpr std::size_t n_modes_print = 8;

    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_rings = n_along + 1;
    const Eigen::Index n_v = mesh.V.rows();
    REQUIRE(n_v == n_rings * n_around);

    auto decompose_and_print = [&](const char* label,
                                   const chladni::shell::ShellModes& modes) {
        std::fprintf(stderr,
            "\n[lme_spurious_decomp] === %s on %dx%d (%lld V) ===\n",
            label, n_around, n_along, static_cast<long long>(n_v));
        const std::size_t n_show = std::min<std::size_t>(
            n_modes_print, static_cast<std::size_t>(modes.omegas.size()));
        for (std::size_t kmode = 0; kmode < n_show; ++kmode) {
            const Eigen::VectorXd u = modes.shapes.col(
                static_cast<Eigen::Index>(kmode));
            const double omega = modes.omegas(
                static_cast<Eigen::Index>(kmode));

            // Decompose into cylindrical components on the (j, i) grid.
            Eigen::MatrixXd ur(n_rings, n_around);
            Eigen::MatrixXd ut(n_rings, n_around);
            Eigen::MatrixXd uz(n_rings, n_around);
            for (Eigen::Index v = 0; v < n_v; ++v) {
                const double th = std::atan2(mesh.V(v, 1), mesh.V(v, 0));
                const double cth = std::cos(th), sth = std::sin(th);
                const double ux = u(3*v+0), uy = u(3*v+1), uzv = u(3*v+2);
                const Eigen::Index j = v / n_around;
                const Eigen::Index i = v % n_around;
                ur(j, i) =  ux * cth + uy * sth;
                ut(j, i) = -ux * sth + uy * cth;
                uz(j, i) = uzv;
            }

            auto fourier_top = [&](const Eigen::MatrixXd& f, const char* name) {
                // m in [0, n_around/2], k axial cos-basis in [0, n_along].
                const Eigen::Index M = n_around / 2;
                const Eigen::Index Kax = n_along;
                Eigen::MatrixXd E = Eigen::MatrixXd::Zero(M + 1, Kax + 1);
                const double total_sq = f.squaredNorm();
                for (Eigen::Index m = 0; m <= M; ++m) {
                    for (Eigen::Index kk = 0; kk <= Kax; ++kk) {
                        double cc = 0.0, sc = 0.0;
                        for (Eigen::Index j = 0; j < n_rings; ++j) {
                            const double az = std::cos(
                                std::numbers::pi * static_cast<double>(kk)
                                * static_cast<double>(j) / static_cast<double>(n_along));
                            for (Eigen::Index i = 0; i < n_around; ++i) {
                                const double th = 2.0 * std::numbers::pi
                                    * static_cast<double>(i)
                                    / static_cast<double>(n_around);
                                const double v_ = f(j, i) * az;
                                cc += v_ * std::cos(static_cast<double>(m) * th);
                                sc += v_ * std::sin(static_cast<double>(m) * th);
                            }
                        }
                        E(m, kk) = cc*cc + sc*sc;   // unnormalised (fractions below)
                    }
                }
                struct Bin { Eigen::Index m, k; double e; };
                std::vector<Bin> top;
                top.reserve(static_cast<std::size_t>((M + 1) * (Kax + 1)));
                for (Eigen::Index m = 0; m <= M; ++m)
                    for (Eigen::Index kk = 0; kk <= Kax; ++kk)
                        top.push_back({m, kk, E(m, kk)});
                std::partial_sort(top.begin(), top.begin() + 3, top.end(),
                    [](const Bin& a, const Bin& b) { return a.e > b.e; });
                const double E_sum = E.sum();
                std::fprintf(stderr,
                    "    %s |.|^2=%.3e  top (m,k,frac-of-bin-energy):",
                    name, total_sq);
                for (int b = 0; b < 3; ++b) {
                    std::fprintf(stderr, "  (%lld,%lld,%.2f)",
                        static_cast<long long>(top[b].m),
                        static_cast<long long>(top[b].k),
                        top[b].e / std::max(E_sum, 1e-30));
                }
                std::fprintf(stderr, "\n");
            };

            const double n_r = ur.norm(), n_t = ut.norm(), n_z = uz.norm();
            const double n_tot = std::sqrt(n_r*n_r + n_t*n_t + n_z*n_z);
            std::fprintf(stderr,
                "  mode %zu  omega=%8.2f rad/s  "
                "|u_r|/|u|=%.2f  |u_th|/|u|=%.2f  |u_z|/|u|=%.2f\n",
                kmode, omega,
                n_r / std::max(n_tot, 1e-30),
                n_t / std::max(n_tot, 1e-30),
                n_z / std::max(n_tot, 1e-30));
            fourier_top(ur, "u_r    ");
            fourier_top(ut, "u_theta");
            fourier_top(uz, "u_z    ");
        }
    };

    {
        chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
        chladni::shell::LMEAssembler a{p};
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, n_modes_print, a);
        decompose_and_print("LME (ghost-free)", m);
    }
    {
        chladni::shell::LoopAssembler a;
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, n_modes_print, a);
        decompose_and_print("Loop", m);
    }

    SUCCEED("LME vs Loop 2D Fourier decomposition printed to stderr");
}

TEST_CASE("cylinder STATIC pinch — LME vs Loop (paper's 'comparable to "
          "subdivision FE' claim)",
          "[.diag][shell][cylinder][static_pinch]")
{
    // Reproduction-style test of Millán 2011's headline claim that LME
    // is "better or comparable than subdivision finite elements" (= Loop
    // here). The paper's benchmarks are STATIC, so we compare LME vs
    // Loop on a static n=2 ovalling pinch — BC-free (self-equilibrated
    // 4-point radial load, rigid modes projected out), which needs no
    // Dirichlet machinery. If LME's compliance ≈ Loop's, the paper's
    // claim holds; if LME is far stiffer, our LME contradicts the paper
    // even statically → a fidelity bug, not just a modal artifact.
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    const auto mesh = chladni::mesh::generate_cylinder(R, L, 24, 32);
    const auto mat  = steel();
    const auto sm   = chladni::shell::shell_material_from_isotropic(mat, h);
    const Eigen::Index n_v = mesh.V.rows();
    const Eigen::Index ndof = 3 * n_v;

    // n=2 ovalling load: radial point loads ~cos(2θ) on the mid-axis
    // ring (self-equilibrated: net force & moment zero by symmetry).
    const double z_mid = 0.5 * L;
    Eigen::VectorXd f = Eigen::VectorXd::Zero(ndof);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        if (std::abs(mesh.V(v, 2) - z_mid) > 0.5 * (L / 32)) continue;
        const double th = std::atan2(mesh.V(v, 1), mesh.V(v, 0));
        const double fr = std::cos(2.0 * th);             // radial amplitude
        f(3 * v + 0) += fr * std::cos(th);
        f(3 * v + 1) += fr * std::sin(th);
    }

    // Six rigid-body modes (ghost-free), orthonormalised.
    Eigen::MatrixXd Rb(ndof, 6);
    Rb.setZero();
    for (Eigen::Index v = 0; v < n_v; ++v) {
        const double x = mesh.V(v, 0), y = mesh.V(v, 1), z = mesh.V(v, 2);
        Rb(3*v+0,0)=1; Rb(3*v+1,1)=1; Rb(3*v+2,2)=1;       // translations
        Rb(3*v+1,3)=-z; Rb(3*v+2,3)= y;                     // rot x
        Rb(3*v+0,4)= z; Rb(3*v+2,4)=-x;                     // rot y
        Rb(3*v+0,5)=-y; Rb(3*v+1,5)= x;                     // rot z
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Rb);
    const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(ndof, 6);
    // Project the load orthogonal to rigid modes (ensure consistency).
    const Eigen::VectorXd f_proj = f - Q * (Q.transpose() * f);

    auto solve_pinch = [&](const char* name,
                           const Eigen::SparseMatrix<double>& K) {
        Eigen::MatrixXd Kd = Eigen::MatrixXd(K);
        const double lam = Kd.diagonal().cwiseAbs().maxCoeff();
        Kd += lam * (Q * Q.transpose());                   // regularise rigid
        const Eigen::VectorXd u = Kd.ldlt().solve(f_proj);
        // Max radial displacement (compliance proxy; bigger = softer).
        double max_ur = 0.0;
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const double th = std::atan2(mesh.V(v,1), mesh.V(v,0));
            const double ur = u(3*v+0)*std::cos(th) + u(3*v+1)*std::sin(th);
            max_ur = std::max(max_ur, std::abs(ur));
        }
        const double compliance = f_proj.dot(u);
        std::fprintf(stderr,
            "[static_pinch] %-5s  max|u_r| = %.4e   compliance fᵀu = %.4e\n",
            name, max_ur, compliance);
        return max_ur;
    };

    double ur_loop = 0.0, ur_lme = 0.0;
    {
        chladni::shell::LoopAssembler a;
        ur_loop = solve_pinch("Loop", a.assemble_K(mesh.V, mesh.F, sm));
    }
    {
        chladni::shell::LMEAssembler::Params p; p.use_ghost_nodes = false;
        chladni::shell::LMEAssembler a{p};
        ur_lme = solve_pinch("LME", a.assemble_K(mesh.V, mesh.F, sm));
    }
    std::fprintf(stderr,
        "[static_pinch] Loop/LME compliance ratio (deflection) = %.1fx  "
        "(paper says LME ~comparable to subdivision FE)\n",
        ur_loop / ur_lme);
    SUCCEED("static pinch LME-vs-Loop printed to stderr");
}

TEST_CASE("free-free cylinder (Loop FEM): convergence under mesh refinement",
          "[shell][modes][cylinder][validation][analytic][loop][convergence]")
{
    // Mirror the SS plate convergence test (L.5c.5) on the cylinder
    // geometry: refine n_around (and proportionally n_along) and pin
    // that the n=4 ovalling error — the most under-resolved at the L.7
    // baseline — drops monotonically as the mesh gets finer. n=2 and
    // n=3 are already very tight at the coarsest resolution, so they
    // are exercised mainly as a sanity check that the omega ordering
    // stays right at every refinement level.
    constexpr double R = 0.10;
    constexpr double L = 2.00;
    constexpr double h = 1.00e-3;
    constexpr std::size_t n_circumferential = 3;     // n = 2, 3, 4

    struct Res { int n_around; int n_along; std::size_t n_modes_solve; };
    constexpr std::array<Res, 3> levels{{
        {16, 24, 30},
        {24, 32, 30},
        {32, 48, 50},   // more axial samples → more axial-bending
                        //   intruders before the n=4 ovalling — bump
                        //   n_modes_solve to keep the slot reachable
    }};

    // Track convergence on n=2 ovalling — the lowest mode and the
    // most cleanly identified by the Fourier-projection heuristic.
    // The higher-n modes (n=3, n=4) develop more axial-bending
    // intruders as n_along grows and the mode-identification picker
    // starts catching the wrong ones; their convergence is exercised
    // by the single-resolution L.7 regression test instead.
    std::array<double, 3> err_n2{};
    for (std::size_t L_i = 0; L_i < levels.size(); ++L_i) {
        const auto r = run_free_free_cylinder_at(
            R, L, h,
            levels[L_i].n_around, levels[L_i].n_along,
            n_circumferential, levels[L_i].n_modes_solve);

        for (std::size_t k = 0; k < n_circumferential; ++k) {
            CAPTURE(L_i, levels[L_i].n_around, levels[L_i].n_along, k);
            REQUIRE(r.fem_omegas[k] > 0.0);
        }
        err_n2[L_i] = r.rel_errs[0];
    }

    INFO("rel_err on n=2 ovalling: "
         << levels[0].n_around << "x" << levels[0].n_along << " -> "
         << err_n2[0] << ",  "
         << levels[1].n_around << "x" << levels[1].n_along << " -> "
         << err_n2[1] << ",  "
         << levels[2].n_around << "x" << levels[2].n_along << " -> "
         << err_n2[2]);

    // Strict monotonic decrease at refinement on the lowest mode.
    REQUIRE(err_n2[1] < err_n2[0]);
    REQUIRE(err_n2[2] < err_n2[1]);

    // Coarse-to-fine improvement of at least 2x going from 16x24 to
    // 32x48 (a 2x linear refinement on n_around). Loop FEM bending
    // is theoretically O(h^2) so we'd expect ~4x; we leave headroom
    // for the aspect-ratio drift in the axial direction.
    REQUIRE(err_n2[0] / err_n2[2] > 2.0);
}
