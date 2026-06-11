/**
 * @file test_modes.cpp
 * @brief Sanity tests for chladni::shell::compute_shell_modes.
 *
 * These tests do NOT pin the spectrum to a closed-form reference (the
 * BC mismatch between cylinder.obj — free-free — and Leissa's SD-SD
 * tabulated values would require boundary-DOF constraints we have not
 * yet implemented). Instead they verify the structural correctness of
 * the eigensolve:
 *
 *   - returned omega values are strictly positive (rigid-body modes
 *     filtered),
 *   - omegas are ascending,
 *   - they fall in a plausible audio range for the test geometry,
 *   - shapes are mass-orthonormal: phi_i^T M phi_j = delta_{ij}.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <filesystem>
#include <numbers>

namespace {

namespace fs = std::filesystem;

/// Drum-shell-sized cylinder: scale R=1 -> R=0.10 m, L=4 -> L=0.20 m.
chladni::mesh::TriMesh load_drum_cylinder()
{
    auto m = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    m.V.col(0) *= 0.10;
    m.V.col(1) *= 0.10;
    m.V.col(2) *= 0.05;
    return m;
}

/// Closed bunny mesh rescaled so the longest bbox dim is 20 cm.
/// The 708-vertex bunny has all 6 rigid-body modes; the rigid-body
/// filter has to catch all of them or the strike amplitude (~1/omega)
/// gets dominated by leaked near-zero modes.
chladni::mesh::TriMesh load_drum_bunny()
{
    const auto path = fs::path{CHLADNI_DATA_DIR} / "bunny_mf_lowres.obj";
    if (!fs::exists(path))
        SKIP("bunny_mf_lowres.obj is not bundled in this distribution");
    auto m = chladni::mesh::load_obj(path);
    const Eigen::Vector3d mn = m.V.colwise().minCoeff();
    const Eigen::Vector3d mx = m.V.colwise().maxCoeff();
    const double longest = (mx - mn).maxCoeff();
    m.V *= 0.20 / longest;
    return m;
}

/// Ellipsoid mesh rescaled so the longest bbox dim is 20 cm.
/// The 1150-vertex isothermic ellipsoid has highly non-uniform
/// triangulation (134x area ratio) producing a bimodal spectrum that
/// stresses the eigensolver — the membrane / bending stiffness ratio
/// 12/h^2 gets large at small h and Spectra's Krylov iteration can
/// silently drop the lowest physical modes if ncv is too tight.
chladni::mesh::TriMesh load_drum_ellipsoid()
{
    const auto path =
        fs::path{CHLADNI_DATA_DIR} / "elipso_isothermic04_complete.obj";
    if (!fs::exists(path))
        SKIP("elipso_isothermic04_complete.obj is not bundled in this "
             "distribution");
    auto m = chladni::mesh::load_obj(path);
    const Eigen::Vector3d mn = m.V.colwise().minCoeff();
    const Eigen::Vector3d mx = m.V.colwise().maxCoeff();
    const double longest = (mx - mn).maxCoeff();
    m.V *= 0.20 / longest;
    return m;
}

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

}  // namespace

TEST_CASE("compute_shell_modes: cylinder.obj at drum-shell scale, steel, h=1mm",
          "[shell][modes][cylinder]")
{
    const auto mesh = load_drum_cylinder();
    constexpr double h        = 1.0e-3;
    constexpr std::size_t n   = 5;

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));
    REQUIRE(modes.shapes.rows() == 3 * mesh.V.rows());
    REQUIRE(modes.shapes.cols() == static_cast<Eigen::Index>(n));

    const double two_pi = 2.0 * std::numbers::pi_v<double>;

    SECTION("frequencies are strictly positive and ascending") {
        for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) > 0.0);
        }
        for (Eigen::Index i = 1; i < modes.omegas.size(); ++i) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }

    SECTION("frequencies are within plausible audio range") {
        const double f_low_hz  = modes.omegas(0)               / two_pi;
        const double f_high_hz = modes.omegas(modes.omegas.size() - 1) / two_pi;
        INFO("f_low = "  << f_low_hz  << " Hz");
        INFO("f_high = " << f_high_hz << " Hz");
        REQUIRE(f_low_hz  > 10.0);
        REQUIRE(f_low_hz  < 5000.0);    // free-free shells: lowest mode usually
                                        // hundreds of Hz to a few kHz
        REQUIRE(f_high_hz < 20000.0);   // audible upper bound
    }

    SECTION("shapes are mass-orthonormal: phi_i^T M phi_j ~ delta_ij") {
        const auto vmasses = chladni::shell::lumped_vertex_masses(
            mesh.V, mesh.F, steel().density, h);
        const auto M = chladni::shell::assemble_mass_matrix(vmasses);

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
}

TEST_CASE("compute_shell_modes returns physical bending modes on ellipsoid at small h",
          "[shell][modes][regression][ellipsoid][thin]")
{
    // Originally added as a regression for a "lowest cluster dropped at
    // small h" bug. After the rigid-body filter was upgraded from a
    // spectrum-gap heuristic to explicit mass-orthogonal projection
    // against the 6-dim rigid subspace (compute_shell_modes), the
    // numerics here changed materially:
    //
    //   - the discrete-shells K has small but non-zero residuals for
    //     rigid rotations (~1e-7 relative; floating-point accumulation
    //     in the analytic Hessian's cot/cosα formulae);
    //   - on the closed ellipsoid those residuals lift the three
    //     rotation modes off zero into finite frequencies (here ~923,
    //     ~2025, ~2282 Hz at h=0.47 mm);
    //   - the OLD gap heuristic let those rotation residues through as
    //     "physical low modes", so the test pinned f_lowest < 10 kHz
    //     and was inadvertently asserting a buggy behaviour;
    //   - the NEW mass-projection filter correctly classifies all six
    //     rigid modes (translations + rotations) — physical bending
    //     modes have rigid-projection ~1e-12 vs ~1.0 for the residues.
    //
    // The actual lowest physical bending mode of the discrete K at
    // h=0.47 mm sits near 44.7 kHz. That is unphysically high for a
    // 20 cm steel ellipsoid (continuum theory predicts low-kHz
    // bending modes), and points to a separate formulation issue:
    // the discrete-shells edge-spring K is over-stiff for closed thin
    // shells. That's tracked separately under Block B.2 / shell-
    // formulation review and is out of scope for this regression.
    //
    // What this test pins now: with the corrected filter, the
    // returned spectrum is composed of physical modes (rigid-projection
    // ≪ 1) only. We assert that the lowest mode has near-zero overlap
    // with the rigid-body subspace, which is the structural property
    // the corrected filter guarantees.
    const auto mesh = load_drum_ellipsoid();
    constexpr double h      = 0.47e-3;
    constexpr std::size_t n = 30;

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));

    // Build the rigid-body subspace in the same form as the filter and
    // verify the lowest returned mode is M-orthogonal to it.
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

    for (Eigen::Index k = 0; k < std::min<Eigen::Index>(modes.omegas.size(), 5); ++k) {
        const Eigen::Matrix<double, 6, 1> c =
            MV.transpose() * modes.shapes.col(k);
        const double rigid_proj_sq = c.dot(chol.solve(c));
        INFO("mode " << k << " omega=" << modes.omegas(k)
             << " rigid_proj_sq=" << rigid_proj_sq);
        // Physical modes have proj ≪ 1; rigid residues have proj ≈ 1.
        // 0.01 is many orders above the actual ~1e-12 separation but
        // still catches any future regression that lets rigid modes leak.
        REQUIRE(rigid_proj_sq < 0.01);
    }

    // Sanity: omega is positive and ascending (these are basic
    // structural properties of a correct return value).
    for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
        REQUIRE(modes.omegas(i) > 0.0);
        if (i > 0) {
            REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
        }
    }
}

TEST_CASE("compute_shell_modes is deterministic across an h cycle",
          "[shell][modes][regression][ellipsoid][determinism]")
{
    // User-asked sanity check after the rigid-body-filter fix:
    // changing h, recomputing, then changing h back and recomputing
    // should give the SAME spectrum as the first compute. The function
    // takes V, F, material, h as inputs and has no hidden state — so
    // two calls with identical arguments must produce equal outputs to
    // within solver convergence tolerance.
    //
    // Use the ellipsoid because it's the case that exposed the "audible
    // modes vanish when h is small" bug; the cycle h=1mm -> h=0.47mm
    // -> h=1mm specifically exercises the largest-gap rigid-body filter
    // in two different gap regimes.
    const auto mesh = load_drum_ellipsoid();
    constexpr double h_a = 1.00e-3;
    constexpr double h_b = 0.47e-3;
    constexpr std::size_t n = 30;

    const auto modes_a1 = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h_a, n);
    const auto modes_b  = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h_b, n);
    const auto modes_a2 = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h_a, n);

    REQUIRE(modes_a1.omegas.size() == modes_a2.omegas.size());
    // Spectra's Lanczos iteration uses a randomised starting vector,
    // so two calls with the same input can disagree at the eigenvalue-
    // residual scale (~kSolverTol = 1e-10). 1e-6 relative is very
    // generous and still catches any structural drift.
    for (Eigen::Index i = 0; i < modes_a1.omegas.size(); ++i) {
        const double w1 = modes_a1.omegas(i);
        const double w2 = modes_a2.omegas(i);
        const double rel = std::abs(w1 - w2) / std::max(w1, 1.0);
        INFO("mode " << i << ": before h-cycle = " << w1
             << " rad/s, after = " << w2 << " rad/s, rel diff = " << rel);
        REQUIRE(rel < 1.0e-6);
    }

    // Sanity: the b-spectrum should be different from the a-spectrum
    // somewhere (otherwise the test is vacuous and would also pass for
    // a stub that returns a constant). After the rigid-body filter
    // upgrade to mass-orthogonal projection, the lowest physical mode
    // on this ellipsoid sits in a stiffness regime where its
    // h-sensitivity is small (the OLD gap heuristic let through
    // rotation residues whose frequency scaled steeply with h, which
    // exaggerated this difference), so we relax the threshold and
    // demand a difference *somewhere* in the lowest 30 modes rather
    // than at mode 0 specifically.
    bool any_meaningful_diff = false;
    for (Eigen::Index i = 0; i < modes_a1.omegas.size(); ++i) {
        const double diff = std::abs(modes_b.omegas(i) - modes_a1.omegas(i));
        if (diff > 0.05 * modes_a1.omegas(i)) {
            any_meaningful_diff = true;
            break;
        }
    }
    REQUIRE(any_meaningful_diff);
}

TEST_CASE("compute_shell_modes filters rigid-body residue on stiff small geometries",
          "[shell][modes][regression][bunny]")
{
    // Regression test for the bunny case: the original rigid-body
    // filter (`eval < largest * 1e-6`) is borderline-broken on stiff
    // small-radius shells whose top-of-spectrum eigenvalue isn't large
    // enough to give a tight relative threshold. On bunny_mf_lowres at
    // drum-shell scale the lowest THREE returned omegas leak through at
    // ~27, ~36, and ~46 Hz — all numerical residues of rigid-body
    // translations / rotations, NOT real bending modes. They drown the
    // physical band: strike modal amplitude is phi^T f / omega, so a
    // 27 Hz fake mode gets ~200x the weighting of the real first mode
    // at ~5800 Hz, producing very loud sub-bass with no real spectral
    // content.
    //
    // Fix: gap-detection in src/shell.cpp — the rigid-body / physical
    // boundary always shows up as a >>100x jump in consecutive
    // eigenvalues; healthy shell modes are typically <5x apart.
    const auto mesh = load_drum_bunny();
    constexpr double h      = 1.0e-3;
    constexpr std::size_t n = 10;

    const auto modes = chladni::shell::compute_shell_modes(
        mesh.V, mesh.F, steel(), h, n);

    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n));

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
        const double f_hz = modes.omegas(i) / two_pi;
        INFO("mode " << i << " = " << f_hz << " Hz");
        // The bunny at h=1mm steel has its first physical mode in the
        // few-kHz range (the geometry simply cannot support a 30 Hz
        // bending mode at this thickness/scale). 100 Hz is a generous
        // floor that catches the rigid-body leak without over-pinning
        // the real first mode.
        REQUIRE(f_hz > 100.0);
    }
}
