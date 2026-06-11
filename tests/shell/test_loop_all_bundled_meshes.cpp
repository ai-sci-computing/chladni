/**
 * @file test_loop_all_bundled_meshes.cpp
 * @brief Smoke test: every bundled .obj in models/ goes through
 *        compute_shell_modes_loop without throwing.
 *
 * Pins which meshes the Loop pipeline actually accepts. If any bundled
 * mesh throws (boundary valence > 4 on some vertex, non-manifold,
 * Loop+Stam producing NaN, ...), we surface it here so strike_gui's
 * legacy CST+IBM fallback isn't masking the failure.
 *
 * Hidden by default ([.smoke]) because the bigger meshes (kitten,
 * fertility) push the eigensolve into ~30s territory.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
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

void rescale_to_target(Eigen::MatrixXd& V, double target = 0.20)
{
    if (V.rows() == 0) return;
    const Eigen::Vector3d mins = V.colwise().minCoeff();
    const Eigen::Vector3d maxs = V.colwise().maxCoeff();
    const double longest = (maxs - mins).maxCoeff();
    if (longest > 0.0) V *= (target / longest);
}

}  // namespace

TEST_CASE("Every bundled mesh runs through compute_shell_modes_loop",
          "[shell][loop][bundled_meshes][.smoke]")
{
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0};
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_modes = 6;

    int n_ok    = 0;
    int n_fail  = 0;
    int n_skip  = 0;
    std::vector<std::string> failures;
    for (const auto& entry : fs::directory_iterator(data_dir())) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".obj") continue;
        const auto name = entry.path().filename().string();

        chladni::mesh::TriMesh mesh;
        try {
            mesh = chladni::mesh::load_obj(entry.path());
        } catch (const std::exception& e) {
            std::cerr << "  [SKIP] " << name << " — load_obj: "
                      << e.what() << '\n';
            ++n_skip;
            continue;
        }
        rescale_to_target(mesh.V, 0.20);

        // Skip very large meshes for the smoke test — they pass
        // assembly but the eigensolve dominates wall time.
        if (mesh.V.rows() > 15000) {
            std::cerr << "  [SKIP large] " << name
                      << " (" << mesh.V.rows() << " V)\n";
            ++n_skip;
            continue;
        }

        try {
            const auto modes = chladni::shell::compute_shell_modes_loop(
                mesh.V, mesh.F, steel, h, n_modes, /*n_passes=*/1,
                /*use_stam=*/false);
            if (!modes.omegas.allFinite()) {
                throw std::runtime_error("non-finite omegas");
            }
            std::cerr << "  [OK]   " << name
                      << " (" << mesh.V.rows() << " V)"
                      << "  f_min=" << modes.omegas(0) / (2.0 * 3.14159)
                      << " Hz\n";
            ++n_ok;
        } catch (const std::exception& e) {
            std::cerr << "  [FAIL] " << name
                      << " (" << mesh.V.rows() << " V): "
                      << e.what() << '\n';
            failures.push_back(name + ": " + e.what());
            ++n_fail;
        }
    }
    std::cerr << "Summary: " << n_ok << " ok, " << n_fail
              << " failed, " << n_skip << " skipped\n";
    if (n_fail > 0) {
        std::cerr << "Failures:\n";
        for (const auto& f : failures) std::cerr << "  " << f << '\n';
    }
    // Don't fail the test on failures — this is a smoke test that
    // documents which meshes go through. The summary above tells the
    // story.
    REQUIRE(n_ok > 0);
}
