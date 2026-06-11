/**
 * @file test_loop_stam_nan_repro.cpp
 * @brief Repro for the Stam-path NaN seen in strike_gui on genus3 @ h=0.28mm.
 *
 * User report (2026-05-12): loading genus3smooth.obj into strike_gui with
 * use_stam=true and h_mm dragged down to 0.28 produced a spectrum starting
 * with NaN. The first mode was NaN, the rest finite. Mesh "disappeared"
 * because update_viz_mesh propagated the NaN to vertex positions.
 *
 * Diagnosis (2026-05-12): K is finite end-to-end (test 1 and test 2 below
 * confirm 0 non-finite entries / 0 bad K_e). The NaN came from
 * solve_modal_eigenproblem_with_rigid_filter calling std::sqrt on a
 * slightly-negative eigenvalue that the 0.5 mass-projection threshold
 * misclassified as physical. Spectra shift-invert near σ = -1 can return
 * a borderline rigid mode with λ ε-negative; mathematically λ ≥ 0 for PSD
 * K + SPD M but floating-point pushes the boundary cluster either way.
 *
 * Fix: also reject λ ≤ 0 in the projection-filter loop. Pinned below by
 * the `[stam][nan_fix]` test which runs the genus3 case end-to-end and
 * requires every omega to be finite (positive).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path data_dir()
{
    if (const char* env = std::getenv("CHLADNI_DATA_DIR")) {
        return fs::path{env};
    }
    return fs::path{CHLADNI_DATA_DIR};
}

// Mirror strike_gui's rescale_to_target_size — applies a uniform scale
// to V so the longest bounding-box axis equals the target. The user's
// genus3 NaN was at 20 cm; without rescaling the raw OBJ is much larger
// and the frequencies don't match the GUI log.
void rescale_to_target_size(Eigen::MatrixXd& V, double target_size_m)
{
    if (V.rows() == 0) return;
    const Eigen::Vector3d mins = V.colwise().minCoeff();
    const Eigen::Vector3d maxs = V.colwise().maxCoeff();
    const double longest = (maxs - mins).maxCoeff();
    if (!std::isfinite(longest) || longest <= 0.0) return;
    V *= (target_size_m / longest);
}

}  // namespace

TEST_CASE("Stam path NaN repro on genus3smooth @ h=0.28mm",
          "[shell][loop][stam][nan][.repro]")
{
    const fs::path mesh_path = data_dir() / "genus3smooth.obj";
    if (!fs::exists(mesh_path)) {
        WARN("Skipping: " << mesh_path << " not found");
        return;
    }
    auto mesh = chladni::mesh::load_obj(mesh_path);
    rescale_to_target_size(mesh.V, 0.20);
    REQUIRE(mesh.V.rows() > 0);

    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0};
    const double h = 0.28e-3;
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    // Build K via the Loop+Stam path with n_passes=1 (the GUI default).
    const auto K = chladni::shell::loop::assemble_stiffness_loop(
        mesh.V, mesh.F, sm, /*n_passes=*/1, /*use_stam=*/true);

    // Scan for non-finite entries in K. If we find any, the bug is
    // upstream of the eigensolve.
    bool any_nan = false;
    Eigen::Index n_nonfinite = 0;
    for (int k = 0; k < K.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                if (!any_nan) {
                    std::cerr << "first non-finite K entry: ("
                              << it.row() << ", " << it.col() << ") = "
                              << it.value() << '\n';
                }
                any_nan = true;
                ++n_nonfinite;
            }
        }
    }
    std::cerr << "K has " << K.nonZeros() << " non-zeros, "
              << n_nonfinite << " non-finite\n";
    REQUIRE_FALSE(any_nan);
}

TEST_CASE("Stam path end-to-end: spectrum on genus3 @ h=0.28mm — no NaN",
          "[shell][loop][stam][nan_fix]")
{
    const fs::path mesh_path = data_dir() / "genus3smooth.obj";
    if (!fs::exists(mesh_path)) {
        WARN("Skipping: " << mesh_path << " not found");
        return;
    }
    auto mesh = chladni::mesh::load_obj(mesh_path);
    rescale_to_target_size(mesh.V, 0.20);

    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0};
    const double h = 0.28e-3;

    const auto modes = chladni::shell::compute_shell_modes_loop(
        mesh.V, mesh.F, steel, h, /*n_modes=*/30,
        /*n_passes=*/1, /*use_stam=*/true);
    std::cerr << "omegas (rad/s): ";
    for (Eigen::Index k = 0; k < std::min<Eigen::Index>(12, modes.omegas.size()); ++k) {
        std::cerr << modes.omegas(k) << " ";
    }
    std::cerr << "\nspectrum (Hz): ";
    for (Eigen::Index k = 0; k < std::min<Eigen::Index>(12, modes.omegas.size()); ++k) {
        std::cerr << (modes.omegas(k) / (2.0 * M_PI)) << " ";
    }
    std::cerr << "\nany non-finite omegas: "
              << (modes.omegas.allFinite() ? "no" : "YES") << "\n";
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        if (!std::isfinite(modes.omegas(k))) {
            std::cerr << "  omega[" << k << "] = " << modes.omegas(k) << "\n";
        }
    }
    REQUIRE(modes.omegas.allFinite());
}

TEST_CASE("Stam path per-element NaN scan on genus3smooth @ h=0.28mm",
          "[shell][loop][stam][nan][.repro]")
{
    // Granular scan: same inputs as above but go through the assembly
    // manually so we can flag WHICH irregular face's K_e first has NaN.
    using namespace chladni::shell::loop;

    const fs::path mesh_path = data_dir() / "genus3smooth.obj";
    if (!fs::exists(mesh_path)) {
        WARN("Skipping: " << mesh_path << " not found");
        return;
    }
    auto mesh = chladni::mesh::load_obj(mesh_path);
    rescale_to_target_size(mesh.V, 0.20);

    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0};
    const double h = 0.28e-3;
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    // Same pipeline as assemble_stiffness_loop: subdivide once, augment,
    // assemble by hand to inspect element K_e values.
    const auto sub = loop_subdivide_n_times(mesh.V, mesh.F, 1);
    const auto aug = augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto patches = build_patch_stencils(aug.V_aug, aug.F_aug);

    std::map<int, StamEvaluator> stam_cache;
    Eigen::Index n_irregular = 0;
    Eigen::Index n_bad = 0;
    int worst_valence = -1;
    double worst_a_det = std::numeric_limits<double>::infinity();
    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        const auto& p = patches[static_cast<std::size_t>(f)];
        int n_irreg = 0;
        int kev = -1;
        for (int k = 0; k < 3; ++k) {
            if (p.corner_valences[static_cast<std::size_t>(k)] != 6) {
                ++n_irreg;
                kev = k;
            }
        }
        if (n_irreg != 1) continue;
        ++n_irregular;
        const int N = p.corner_valences[static_cast<std::size_t>(kev)];

        std::vector<Eigen::Index> dofs;
        try {
            dofs = gather_stam_patch_dofs(p, aug.F_aug);
        } catch (const std::exception& e) {
            std::cerr << "gather throw on face " << f << " (N=" << N
                      << "): " << e.what() << '\n';
            continue;
        }
        auto it = stam_cache.find(N);
        if (it == stam_cache.end()) {
            it = stam_cache.emplace(N, make_stam_evaluator(N)).first;
        }
        const StamEvaluator& ev = it->second;
        StamPatchEvaluation pe;
        try {
            pe = evaluate_patch_stam(ev, dofs, aug.V_aug, 1.0/3.0, 1.0/3.0);
        } catch (const std::exception& e) {
            std::cerr << "evaluate_patch_stam throw on face " << f
                      << " (N=" << N << "): " << e.what() << '\n';
            continue;
        }
        const Eigen::Vector3d a1 = pe.cov_basis.col(0);
        const Eigen::Vector3d a2 = pe.cov_basis.col(1);
        const double a_det =
            a1.dot(a1) * a2.dot(a2) - (a1.dot(a2)) * (a1.dot(a2));
        if (a_det < worst_a_det) worst_a_det = a_det;
        Eigen::MatrixXd K_e;
        try {
            K_e = element_stiffness_stam(ev, dofs, aug.V_aug, sm);
        } catch (const std::exception& e) {
            std::cerr << "element_stiffness_stam throw on face " << f
                      << " (N=" << N << "): a_det=" << a_det
                      << "  " << e.what() << '\n';
            continue;
        }
        if (!K_e.allFinite()) {
            ++n_bad;
            if (n_bad <= 5) {
                std::cerr << "NaN K_e on face " << f << " (N=" << N
                          << "), a_det=" << a_det
                          << ", K_e.norm=" << K_e.norm()
                          << ", max|K_e|=" << K_e.cwiseAbs().maxCoeff() << '\n';
            }
            if (worst_valence < 0) worst_valence = N;
        }
    }
    std::cerr << "irregular faces seen: " << n_irregular
              << "  bad (NaN K_e): " << n_bad
              << "  worst a_det: " << worst_a_det
              << "  first bad valence: " << worst_valence << '\n';
    REQUIRE(n_bad == 0);
}
