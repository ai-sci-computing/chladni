/**
 * @file test_modes_loop.cpp
 * @brief Structural sanity tests for chladni::shell::compute_shell_modes_loop.
 *
 * Loop-subdivision shell modal solve (L.6 wiring). These tests do NOT
 * pin the spectrum to closed-form references — that is L.7's job, gated
 * under @c [.skip] in
 * @c test_modes_vs_free_free_cylinder_analytic.cpp until the threshold
 * tightening. They verify the structural correctness of the Loop modal
 * pipeline, mirroring the parallel sanity tests for the legacy CST + IBM
 * @ref chladni::shell::compute_shell_modes in @c test_modes.cpp:
 *
 *   - returned omega values are strictly positive (rigid-body modes
 *     filtered),
 *   - omegas are ascending,
 *   - shapes are mass-orthonormal: @f$ \phi_i^\top M \phi_j = \delta_{ij} @f$,
 *   - the rigid-body filter keeps the lowest 6 zero / near-zero modes
 *     out of the returned spectrum (verified by checking that the
 *     returned modes have negligible projection onto the rigid-body
 *     subspace).
 *
 * Geometry: small generated cylinder (n_around = 8, n_along = 4) at
 * drum-shell scale, steel, h = 1 mm. Cylinder rim vertices are valence-4
 * (the case @ref chladni::shell::loop::augment_for_loop_boundary
 * supports); all interior vertices are valence-6 and so each interior
 * triangle is a regular Loop patch.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>

namespace {

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

/// Drum-shell-sized generated cylinder: R = 10 cm, L = 20 cm, with a
/// valence-4 boundary (rim) and valence-6 interior, which is exactly
/// the regularity the L.5 augmentation handles.
chladni::mesh::TriMesh make_drum_cylinder()
{
    return chladni::mesh::generate_cylinder(0.10, 0.20, 8, 4);
}

/// Closed octahedron-derived mesh: the bare octahedron has only 6
/// vertices and the eigensolver's rigid-body slack pushes the system
/// past its DOF budget, so we pre-subdivide once externally to give
/// the modal solve a 18-vertex / 32-face mesh to chew on. The 6
/// original octahedron vertices retain valence 4 in the subdivided
/// mesh (Loop subdivision preserves valence at original-vertex
/// slots) so each contains an extraordinary interior vertex that
/// triggers the L.3.4 subdivision branch inside
/// @ref chladni::shell::loop::assemble_stiffness_loop. Effective
/// subdivision depth = 2 (one external + one internal); the residual
/// irregular sub-triangles after the 2nd step have area
/// 1/16 of an octahedron face, totalling ~19% of the surface
/// dropped under the one-pass approximation.
chladni::mesh::TriMesh make_octahedron_subdivided_once()
{
    chladni::mesh::TriMesh raw;
    raw.V.resize(6, 3);
    raw.V <<  1,  0,  0,
             -1,  0,  0,
              0,  1,  0,
              0, -1,  0,
              0,  0,  1,
              0,  0, -1;
    raw.F.resize(8, 3);
    raw.F << 0, 2, 4,
             2, 1, 4,
             1, 3, 4,
             3, 0, 4,
             2, 0, 5,
             1, 2, 5,
             3, 1, 5,
             0, 3, 5;
    const auto sub =
        chladni::shell::loop::loop_subdivide_one_step(raw.V, raw.F);
    chladni::mesh::TriMesh m;
    m.V = sub.V_sub;
    m.F = sub.F_sub;
    return m;
}

}  // namespace

TEST_CASE("compute_shell_modes_loop: cylinder, steel, h=1mm — basic spectrum properties",
          "[shell][loop][modes][cylinder]")
{
    const auto mesh = make_drum_cylinder();
    constexpr double h      = 1.0e-3;
    constexpr std::size_t n = 5;

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));
    REQUIRE(modes.shapes.rows() == 3 * mesh.V.rows());
    REQUIRE(modes.shapes.cols() == static_cast<Eigen::Index>(n));

    SECTION("frequencies are strictly positive and ascending") {
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) > 0.0);
        }
        for (Eigen::Index i = 1; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }

    SECTION("shapes are mass-orthonormal: phi_i^T M phi_j ~ delta_ij") {
        // Use the same M the solver normalizes against. The
        // compute_shell_modes_loop shim defaults to n_passes=1,
        // use_stam=false; mirror those Params here so K and M come from
        // the same LoopAssembler the solver used.
        chladni::shell::LoopAssembler::Params p;
        p.n_passes = 1;
        p.use_stam = false;
        const chladni::shell::LoopAssembler assembler{p};
        const auto M = assembler.assemble_M(
            mesh.V, mesh.F, steel().density * h);

        const Eigen::MatrixXd PhiTMPhi =
            modes.shapes.transpose() * M * modes.shapes;

        for (Eigen::Index i = 0; i < PhiTMPhi.rows(); ++i) {
            for (Eigen::Index j = 0; j < PhiTMPhi.cols(); ++j) {
                const double expected = (i == j) ? 1.0 : 0.0;
                INFO("PhiTMPhi(" << i << "," << j << ") = " << PhiTMPhi(i, j));
                REQUIRE(std::abs(PhiTMPhi(i, j) - expected) < 1e-6);
            }
        }
    }

    SECTION("returned modes have negligible projection onto rigid-body subspace") {
        // Build the same 6-dim rigid-body subspace the filter inside
        // compute_shell_modes_loop uses, and verify each returned mode
        // is M-orthogonal to it. Physical bending modes have projection
        // ≪ 1; rigid residues have projection ≈ 1.
        Eigen::MatrixXd V_rigid = Eigen::MatrixXd::Zero(3 * mesh.V.rows(), 6);
        for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
            const double x = mesh.V(i, 0);
            const double y = mesh.V(i, 1);
            const double z = mesh.V(i, 2);
            V_rigid(3 * i + 0, 0) = 1.0;
            V_rigid(3 * i + 1, 1) = 1.0;
            V_rigid(3 * i + 2, 2) = 1.0;
            V_rigid(3 * i + 1, 3) = -z;
            V_rigid(3 * i + 2, 3) =  y;
            V_rigid(3 * i + 0, 4) =  z;
            V_rigid(3 * i + 2, 4) = -x;
            V_rigid(3 * i + 0, 5) = -y;
            V_rigid(3 * i + 1, 5) =  x;
        }
        const auto vmasses = chladni::shell::lumped_vertex_masses(
            mesh.V, mesh.F, steel().density, h);
        const auto M = chladni::shell::assemble_mass_matrix(vmasses);
        const Eigen::MatrixXd MV = M * V_rigid;
        const Eigen::Matrix<double, 6, 6> G = V_rigid.transpose() * MV;
        const Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(G);

        for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
            const Eigen::Matrix<double, 6, 1> c =
                MV.transpose() * modes.shapes.col(k);
            const double rigid_proj_sq = c.dot(chol.solve(c));
            INFO("mode " << k << " omega=" << modes.omegas(k)
                 << " rigid_proj_sq=" << rigid_proj_sq);
            REQUIRE(rigid_proj_sq < 0.01);
        }
    }
}

TEST_CASE("compute_shell_modes_loop: tiny mesh routes through the dense "
          "eigensolver fallback (R9)",
          "[shell][loop][modes][dense_fallback]")
{
    // icosphere k=1 has 42 vertices = 126 DOF, below the 300-DOF dense
    // threshold, so the eigensolve takes the direct dense generalised path
    // rather than multi-seed shift-invert (least reliable when the requested
    // window approaches the full dimension). This guards that the fallback
    // produces a clean, deterministic physical spectrum.
    const auto mesh = chladni::mesh::generate_icosphere(
        /*radius=*/0.10, /*n_subdivisions=*/1);
    REQUIRE(3 * mesh.V.rows() < 300);

    constexpr double h      = 1.0e-3;
    constexpr std::size_t n = 8;

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));
    REQUIRE(modes.shapes.cols() == static_cast<Eigen::Index>(n));

    SECTION("spectrum is finite, strictly positive and ascending") {
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            REQUIRE(std::isfinite(modes.omegas(i)));
            REQUIRE(modes.omegas(i) > 0.0);
        }
        for (Eigen::Index i = 1; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }

    SECTION("shapes are mass-orthonormal: phi_i^T M phi_j ~ delta_ij") {
        chladni::shell::LoopAssembler::Params p;
        p.n_passes = 1;
        p.use_stam = false;
        const chladni::shell::LoopAssembler assembler{p};
        const auto M = assembler.assemble_M(
            mesh.V, mesh.F, steel().density * h);
        const Eigen::MatrixXd PhiTMPhi =
            modes.shapes.transpose() * M * modes.shapes;
        for (Eigen::Index i = 0; i < PhiTMPhi.rows(); ++i) {
            for (Eigen::Index j = 0; j < PhiTMPhi.cols(); ++j) {
                const double expected = (i == j) ? 1.0 : 0.0;
                INFO("PhiTMPhi(" << i << "," << j << ") = " << PhiTMPhi(i, j));
                REQUIRE(std::abs(PhiTMPhi(i, j) - expected) < 1e-6);
            }
        }
    }

    SECTION("dense path is deterministic (no random seeds)") {
        // The dense solver has no seeded starting vectors, so a second call
        // must reproduce the spectrum bit-for-bit — unlike the multi-seed
        // shift-invert path it replaces for tiny systems.
        const auto modes2 = chladni::shell::compute_shell_modes_loop(
            mesh.V, mesh.F, steel(), h, n);
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            REQUIRE(modes2.omegas(i) == modes.omegas(i));
        }
    }
}

TEST_CASE("compute_shell_modes_loop: invalid args throw",
          "[shell][loop][modes][validation]")
{
    const auto mesh = make_drum_cylinder();

    SECTION("n_modes == 0") {
        REQUIRE_THROWS_AS(
            chladni::shell::compute_shell_modes_loop(
                mesh.V, mesh.F, steel(), 1.0e-3, 0),
            std::invalid_argument);
    }
    SECTION("non-positive thickness") {
        REQUIRE_THROWS_AS(
            chladni::shell::compute_shell_modes_loop(
                mesh.V, mesh.F, steel(), 0.0, 3),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            chladni::shell::compute_shell_modes_loop(
                mesh.V, mesh.F, steel(), -1.0, 3),
            std::invalid_argument);
    }
}

TEST_CASE("Loop cylinder mode-shape diagnostic (run explicitly: ./chladni_tests \"Loop cylinder mode-shape*\")",
          "[shell][loop][diagnostic][.skip]")
{
    // L.7 hypothesis-test diagnostic: decompose each Loop mode's
    // displacement field on a generated cylinder into
    // (a) circumferential energy per Fourier wavenumber n,
    // (b) axial-mass-weighted energy per ring j,
    // and print the results so we can identify which mode is which
    // (n=2/3/4 ovalling vs axial-bending intruders).
    //
    // Cylinder convention from chladni::mesh::generate_cylinder:
    //  vertex (i, j) is at (R cos φ_i, R sin φ_i, z_j)
    //  with φ_i = 2π i / n_around, z_j = L j / n_along,
    //  flattened ID = j * n_around + i.
    //
    // Radial displacement at vertex (i, j):
    //   u_r(i, j) = u_x cos φ_i + u_y sin φ_i.
    // Circumferential cosine/sine coefficients at ring j:
    //   c_n(j) = (2 / n_around) Σ_i u_r(i, j) cos(n φ_i)
    //   s_n(j) = (2 / n_around) Σ_i u_r(i, j) sin(n φ_i)
    // Energy per circumferential wavenumber, summed over rings:
    //   E_n = Σ_j (c_n(j)² + s_n(j)²)
    constexpr double R       = 0.10;
    constexpr double L       = 2.00;
    constexpr double h       = 1.0e-3;
    constexpr int    n_around = 24;
    constexpr int    n_along  = 32;

    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    constexpr std::size_t n_modes_dump = 24;
    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n_modes_dump);

    constexpr int n_max     = 6;
    constexpr int n_rings   = n_along + 1;
    const double  two_pi    = 2.0 * std::numbers::pi_v<double>;

    // Cache φ_i and (cos, sin) tables.
    std::vector<double> cos_phi(n_around), sin_phi(n_around);
    for (int i = 0; i < n_around; ++i) {
        const double phi = two_pi * static_cast<double>(i) / n_around;
        cos_phi[i] = std::cos(phi);
        sin_phi[i] = std::sin(phi);
    }

    auto circumferential_energy = [&](const Eigen::VectorXd& u, int n)
    {
        // Returns (E_n, j_argmax_E_per_ring, axial_distribution_string).
        std::vector<double> per_ring(n_rings, 0.0);
        for (int j = 0; j < n_rings; ++j) {
            double c = 0.0, s = 0.0;
            for (int i = 0; i < n_around; ++i) {
                const Eigen::Index v = j * n_around + i;
                const double ux = u(3 * v + 0);
                const double uy = u(3 * v + 1);
                const double u_r = ux * cos_phi[i] + uy * sin_phi[i];
                const double cn  = std::cos(n * two_pi * i / n_around);
                const double sn  = std::sin(n * two_pi * i / n_around);
                c += u_r * cn;
                s += u_r * sn;
            }
            c *= 2.0 / n_around;
            s *= 2.0 / n_around;
            per_ring[j] = c * c + s * s;
        }
        double total = 0.0;
        for (double e : per_ring) total += e;
        return std::make_pair(total, per_ring);
    };

    std::cerr << "\n=== Loop cylinder mode diagnostic ===\n";
    std::cerr << "Geometry: R=" << R << " L=" << L << " h=" << h
              << " n_around=" << n_around << " n_along=" << n_along << "\n";
    std::cerr << "Mode | omega(rad/s) | E_n=0   E_n=1   E_n=2   E_n=3   E_n=4   E_n=5   E_n=6 | dom_n | axial\n";
    for (std::size_t k = 0; k < n_modes_dump; ++k) {
        const Eigen::VectorXd u = modes.shapes.col(static_cast<Eigen::Index>(k));
        std::vector<double> Es(n_max + 1, 0.0);
        std::vector<std::vector<double>> per_ring_all(n_max + 1);
        for (int n = 0; n <= n_max; ++n) {
            auto [E, per_ring] = circumferential_energy(u, n);
            Es[n] = E;
            per_ring_all[n] = std::move(per_ring);
        }
        // Find dominant n.
        int dom_n = 0;
        for (int n = 1; n <= n_max; ++n) if (Es[n] > Es[dom_n]) dom_n = n;
        // Build a compact axial distribution string for the dominant n:
        // "F" (flat — ovalling, no axial wave), "1H" (one half-wave), etc.
        // Simply look at per_ring_all[dom_n] and see if it's constant or
        // has nodes (sign changes after taking sqrt with a phase). Too
        // fancy for now — just print the first / middle / last values
        // for visual inspection.
        const auto& pr = per_ring_all[dom_n];
        const double e0 = pr.front();
        const double e_mid = pr[n_along / 2];
        const double e_end = pr.back();
        std::cerr.setf(std::ios::scientific, std::ios::floatfield);
        std::cerr.precision(2);
        std::cerr << "  " << k
                  << " | " << modes.omegas(static_cast<Eigen::Index>(k))
                  << " | ";
        for (int n = 0; n <= n_max; ++n) {
            std::cerr << Es[n] << "  ";
        }
        std::cerr << "| " << dom_n << " | "
                  << "j=0:" << e0 << "  mid:" << e_mid << "  end:" << e_end
                  << "\n";
    }
    std::cerr << "=== end diagnostic ===\n\n";
    SUCCEED();  // no assertion — informational test only
}

TEST_CASE("compute_shell_modes_loop: closed octahedron — irregular L.3.4 path",
          "[shell][loop][modes][closed][irregular]")
{
    // Closed mesh derived from a once-subdivided octahedron — the 6
    // original octahedron vertices retain valence 4 and trigger the
    // L.3.4 subdivision branch inside compute_shell_modes_loop. The
    // mesh is still far too coarse to give physically meaningful
    // frequencies, so this test pins only the structural properties.
    // The numerical value of the lowest omega depends on the (under-
    // resolved) mesh and on the one-pass approximation; we don't
    // assert against any analytic reference.
    const auto mesh = make_octahedron_subdivided_once();
    constexpr double h      = 1.0e-3;
    constexpr std::size_t n = 3;

    REQUIRE_NOTHROW(chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n));

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));
    REQUIRE(modes.shapes.rows() == 3 * mesh.V.rows());
    REQUIRE(modes.shapes.cols() == static_cast<Eigen::Index>(n));

    SECTION("frequencies are strictly positive and ascending") {
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) > 0.0);
        }
        for (Eigen::Index i = 1; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }

    SECTION("shapes are mass-orthonormal: phi_i^T M phi_j ~ delta_ij") {
        // Mirror the shim's defaults (n_passes=1, use_stam=false) so K
        // and M come from the same LoopAssembler the solver used.
        chladni::shell::LoopAssembler::Params p;
        p.n_passes = 1;
        p.use_stam = false;
        const chladni::shell::LoopAssembler assembler{p};
        const auto M = assembler.assemble_M(
            mesh.V, mesh.F, steel().density * h);
        const Eigen::MatrixXd PhiTMPhi =
            modes.shapes.transpose() * M * modes.shapes;
        for (Eigen::Index i = 0; i < PhiTMPhi.rows(); ++i) {
            for (Eigen::Index j = 0; j < PhiTMPhi.cols(); ++j) {
                const double expected = (i == j) ? 1.0 : 0.0;
                INFO("PhiTMPhi(" << i << "," << j << ") = " << PhiTMPhi(i, j));
                REQUIRE(std::abs(PhiTMPhi(i, j) - expected) < 1e-6);
            }
        }
    }

    SECTION("returned modes have negligible projection onto rigid-body subspace") {
        // 6-dim rigid-body subspace (3 translations + 3 rotations
        // about the origin, which is the octahedron's centre). Each
        // returned non-rigid mode must have negligible mass-weighted
        // projection on this subspace.
        Eigen::MatrixXd V_rigid = Eigen::MatrixXd::Zero(3 * mesh.V.rows(), 6);
        for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
            const double x = mesh.V(i, 0);
            const double y = mesh.V(i, 1);
            const double z = mesh.V(i, 2);
            V_rigid(3 * i + 0, 0) = 1.0;
            V_rigid(3 * i + 1, 1) = 1.0;
            V_rigid(3 * i + 2, 2) = 1.0;
            V_rigid(3 * i + 1, 3) = -z;
            V_rigid(3 * i + 2, 3) =  y;
            V_rigid(3 * i + 0, 4) =  z;
            V_rigid(3 * i + 2, 4) = -x;
            V_rigid(3 * i + 0, 5) = -y;
            V_rigid(3 * i + 1, 5) =  x;
        }
        const auto vmasses = chladni::shell::lumped_vertex_masses(
            mesh.V, mesh.F, steel().density, h);
        const auto M = chladni::shell::assemble_mass_matrix(vmasses);
        const Eigen::MatrixXd MV = M * V_rigid;
        const Eigen::Matrix<double, 6, 6> G = V_rigid.transpose() * MV;
        const Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(G);

        for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
            const Eigen::Matrix<double, 6, 1> c =
                MV.transpose() * modes.shapes.col(k);
            const double rigid_proj_sq = c.dot(chol.solve(c));
            INFO("mode " << k << " omega=" << modes.omegas(k)
                 << " rigid_proj_sq=" << rigid_proj_sq);
            REQUIRE(rigid_proj_sq < 0.01);
        }
    }
}

TEST_CASE("compute_shell_modes_loop: closed icosphere multi-pass converges (n_passes ∈ {1, 2, 3})",
          "[shell][loop][modes][closed][irregular][multipass][convergence][icosphere]")
{
    // L.3.4 multi-pass on a meaningful closed mesh. The icosphere at
    // n_subdivisions=1 has 42 vertices (the bare icosahedron is too
    // small for the eigensolver's rigid-body slack of 16, and at
    // n_subdivisions=0 the Cirak-Ortiz §4.6 step-1 prerequisite is
    // violated anyway because every face has 3 valence-5 corners).
    //
    // Run with n_passes = 1, 2, 3 and verify the lowest non-rigid
    // omega converges geometrically: each extra pass shrinks the
    // dropped irregular-area residual by 1/4, so K stiffens by a
    // shrinking amount per pass and omega increases toward a limit.
    const auto mesh = chladni::mesh::generate_icosphere(0.10, 1);
    constexpr double      h = 1.0e-3;
    constexpr std::size_t n = 3;

    std::array<double, 3> omega_lowest{};
    for (int p = 1; p <= 3; ++p) {
        const auto modes = chladni::shell::compute_shell_modes_loop(
            mesh.V, mesh.F, steel(), h, n, /*n_passes=*/p);
        REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            CAPTURE(p, i);
            REQUIRE(modes.omegas(i) > 0.0);
        }
        omega_lowest[static_cast<std::size_t>(p - 1)] = modes.omegas(0);
    }

    INFO("icosphere lowest omega: n_passes=1 -> " << omega_lowest[0]
         << ",  n_passes=2 -> " << omega_lowest[1]
         << ",  n_passes=3 -> " << omega_lowest[2]);

    // Monotonic increase: more passes pull more bending energy back
    // into K and stiffen the lowest mode.
    REQUIRE(omega_lowest[1] > omega_lowest[0]);
    REQUIRE(omega_lowest[2] > omega_lowest[1]);

    // Geometric convergence: pass-to-pass increment shrinks. The
    // theoretical ratio is 1/4 per pass; we leave headroom for the
    // mode-identification heuristic (lowest mode here may shift
    // identity slightly across resolutions on a sphere where many
    // modes are near-degenerate).
    const double d12 = omega_lowest[1] - omega_lowest[0];
    const double d23 = omega_lowest[2] - omega_lowest[1];
    REQUIRE(d23 < d12);
}

TEST_CASE("compute_shell_modes_loop: closed octahedron multi-pass converges (n_passes ∈ {1, 2, 3})",
          "[shell][loop][modes][closed][irregular][multipass][convergence]")
{
    // L.3.4 multi-pass exercise: re-run the closed octahedron fixture
    // with n_passes = 1, 2, 3 and verify the lowest non-rigid omega
    // converges (each extra pass shrinks the dropped irregular-area
    // residual by 1/4). For a Cirak-Ortiz §4.6 step-1 underestimate
    // (some triangles' bending energy dropped), additional passes
    // pull more energy back into K — but with consistent mass M
    // *also* refines with n_passes (M is assembled from the same
    // multi-pass kernel), so the ω = √(K/M) trajectory may overshoot
    // the limit and oscillate rather than rising monotonically. The
    // robust convergence pin is on the *magnitude* of the increment,
    // not its sign: |Δ_2to3| should be smaller than |Δ_1to2| by the
    // 1/4 geometric factor (with headroom).
    const auto mesh = make_octahedron_subdivided_once();
    constexpr double      h = 1.0e-3;
    constexpr std::size_t n = 3;

    std::array<double, 3> omega_lowest{};
    for (int p = 1; p <= 3; ++p) {
        const auto modes =
            chladni::shell::compute_shell_modes_loop(
                mesh.V, mesh.F, steel(), h, n, /*n_passes=*/p);
        REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));

        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            CAPTURE(p, i);
            REQUIRE(modes.omegas(i) > 0.0);
        }
        omega_lowest[static_cast<std::size_t>(p - 1)] = modes.omegas(0);
    }

    INFO("lowest omega: n_passes=1 -> " << omega_lowest[0]
         << ",  n_passes=2 -> " << omega_lowest[1]
         << ",  n_passes=3 -> " << omega_lowest[2]);

    // Geometric magnitude convergence: |Δ_2to3| < |Δ_1to2|. The
    // theoretical ratio is 1/4 per pass; we leave headroom for the
    // mode-identification heuristic (lowest mode here may shift
    // identity slightly across resolutions on a sphere where many
    // modes are near-degenerate).
    const double d12 = std::abs(omega_lowest[1] - omega_lowest[0]);
    const double d23 = std::abs(omega_lowest[2] - omega_lowest[1]);
    REQUIRE(d23 < d12);

    // Sanity: all three passes should land within a tight relative
    // window of each other. Under-converged passes can sit ~1 % off
    // the limit; require the three-way spread to stay below 5 %.
    const double spread =
        (*std::max_element(omega_lowest.begin(), omega_lowest.end())
         - *std::min_element(omega_lowest.begin(), omega_lowest.end()))
        / omega_lowest[0];
    REQUIRE(spread < 0.05);
}

TEST_CASE("compute_shell_modes_loop: explicit ShellMaterial overload yields the same spectrum as the isotropic path",
          "[shell][loop][modes][overload]")
{
    // Pins the API contract of the explicit ShellMaterial overload:
    // passing the same material that shell_material_from_isotropic
    // derives must reproduce the IsotropicMaterial-only path's modes
    // exactly. Lets callers override k_L / k_B independently of the
    // isotropic material later, without changing the default behaviour.
    const auto mesh = make_drum_cylinder();
    constexpr double h = 1.0e-3;
    constexpr std::size_t n = 3;

    const auto sm = chladni::shell::shell_material_from_isotropic(steel(), h);

    const auto modes_iso = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h, n);
    const auto modes_sm  = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), sm, h, n);

    REQUIRE(modes_iso.omegas.size() == modes_sm.omegas.size());
    for (Eigen::Index i = 0; i < modes_iso.omegas.size(); ++i) {
        const double rel =
            std::abs(modes_iso.omegas(i) - modes_sm.omegas(i))
            / std::max(modes_iso.omegas(i), 1.0);
        REQUIRE(rel < 1e-6);
    }
}
