/**
 * @file main.cpp
 * @brief synth_demo — compute the modes of a real cylindrical shell
 *        (cylinder.obj scaled to drum-shell dimensions, steel, 1 mm thick)
 *        and write the resulting impulse response to a WAV file.
 *
 * Pipeline:
 *   1. Load cylinder.obj from CHLADNI_DATA_DIR (else ./models/).
 *   2. Scale R=1, L=4 to user-chosen drum-shell dimensions
 *      (defaults: R=0.10 m, L=0.20 m -> roughly a tom shell).
 *   3. compute_shell_modes -> 30 lowest free-free shell modes.
 *   4. Apply Rayleigh damping d_i = (alpha + beta omega_i^2) / 2.
 *   5. Hammer impulse: project a unit force at one vertex onto each mode
 *      via phi_i (uniform-strike approximation: sum over rim vertices).
 *      A high-mode roll-off (1/sqrt(1 + (omega/omega_c)^2)) approximates
 *      a finite-contact-patch hammer.
 *   6. synthesize_resonator_bank -> 2.5 s mono float buffer.
 *   7. Normalise to 0.85 peak; write_mono16 -> WAV.
 *
 * Usage:
 *     synth_demo [output.wav]
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/synth.hpp>
#include <chladni/wav.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// Resolve cylinder.obj from CHLADNI_DATA_DIR if defined, else from
/// the current working directory's models/.
fs::path default_mesh_path()
{
#ifdef CHLADNI_DATA_DIR
    return fs::path{CHLADNI_DATA_DIR} / "cylinder.obj";
#else
    return fs::path{"models"} / "cylinder.obj";
#endif
}

}  // namespace

int main(int argc, char** argv)
{
    namespace cmesh = chladni::mesh;
    namespace csyn  = chladni::synth;
    namespace cwav  = chladni::wav;
    namespace csh   = chladni::shell;

    // ---- Geometry: scale cylinder.obj (R=1, L=4) to drum-shell dimensions.
    constexpr double R   = 0.10;       // 0.10 m radius (~ 8" tom)
    constexpr double L_z = 0.20;       // 0.20 m height
    constexpr double h   = 1.0e-3;     // 1 mm wall thickness

    // ---- Material: mild steel.
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0,
    };

    constexpr std::size_t n_modes      = 30;
    constexpr double      sample_rate  = 44100.0;
    constexpr double      duration_s   = 2.5;

    // Rayleigh damping: d_i = (alpha + beta * omega_i^2) / 2. Tuned for a
    // shell that rings ~1 s on the fundamental with high partials decaying
    // within a few hundred ms.
    constexpr double alpha = 4.0;     // 1/s, mass-proportional
    constexpr double beta  = 1.0e-7;  // s, stiffness-proportional

    // High-mode roll-off (finite hammer contact): 1 / sqrt(1 + (omega/omega_c)^2)
    constexpr double omega_cutoff = 2.0 * std::numbers::pi_v<double> * 4000.0;

    // ---- Load and scale the mesh.
    const auto mesh_path = default_mesh_path();
    std::cout << "synth_demo: loading " << mesh_path.string() << "\n";
    auto mesh = cmesh::load_obj(mesh_path);
    mesh.V.col(0) *= R;
    mesh.V.col(1) *= R;
    mesh.V.col(2) *= L_z / 4.0;  // original z extent is [0, 4]

    std::cout << "synth_demo: " << mesh.V.rows() << " vertices, "
              << mesh.F.rows() << " triangles. Solving for "
              << n_modes << " modes... " << std::flush;

    // ---- Eigensolve.
    const auto modes = csh::compute_shell_modes(mesh.V, mesh.F, steel, h, n_modes);
    std::cout << "done.\n";

    std::cout << "spectrum (Hz):";
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        std::cout << " " << modes.omegas(k) / two_pi;
    }
    std::cout << "\n";

    // ---- Build the resonator bank from the modes.
    // Per-mode amplitude approximates a uniform-strike + soft-mallet:
    // amp(omega) = (1/n_modes) * 1/sqrt(1 + (omega/omega_c)^2).
    // (A real strike will eventually project an impulse vector onto
    // each mode shape — that lands with the GUI strike pipeline.)
    std::vector<csyn::ResonatorMode> bank;
    bank.reserve(static_cast<std::size_t>(modes.omegas.size()));
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        const double w = modes.omegas(k);
        const double d = 0.5 * (alpha + beta * w * w);
        const double a = (1.0 / static_cast<double>(modes.omegas.size()))
                       / std::sqrt(1.0 + (w * w) / (omega_cutoff * omega_cutoff));
        bank.push_back({.angular_frequency = w,
                        .damping_rate      = d,
                        .amplitude         = a,
                        .phase             = 0.0});
    }

    auto y = csyn::synthesize_resonator_bank(
        std::span<const csyn::ResonatorMode>{bank}, sample_rate, duration_s);

    // ---- Normalise to 0.85 peak.
    float peak = 0.0F;
    for (float s : y) peak = std::max(peak, std::abs(s));
    if (peak > 0.0F) {
        const float scale = 0.85F / peak;
        for (float& s : y) s *= scale;
    }

    const std::string out_path = (argc > 1) ? argv[1] : "synth_demo.wav";
    try {
        cwav::write_mono16(out_path,
                           std::span<const float>{y},
                           static_cast<std::uint32_t>(sample_rate));
    } catch (const std::exception& e) {
        std::cerr << "synth_demo: failed to write WAV: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "wrote " << y.size() << " samples (" << duration_s << " s) "
              << "to " << out_path << "\n";
    return EXIT_SUCCESS;
}
