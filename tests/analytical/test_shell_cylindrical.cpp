/**
 * @file test_shell_cylindrical.cpp
 * @brief Phase-2 gating test: simply-supported (SD-SD) thin cylindrical
 *        shell under Donnell-Mushtari theory.
 *
 * Pins
 * @ref chladni::analytical::simply_supported_cylindrical_shell_donnell_mushtari_angular_frequencies
 * to reference values computed independently (NumPy companion-matrix
 * solve of the Donnell-Mushtari cubic, Leissa 1973 eq. 2.35-2.36).
 *
 * Test shell:
 *   R = 0.10 m, L = 0.20 m, h = 1 mm, steel
 *   (E = 200 GPa, nu = 0.30, rho = 7850 kg/m^3).
 *
 * Lowest five modes (Hz; m, n in parentheses):
 *    962.2805   (1, 5)
 *   1052.0696   (1, 6)
 *   1124.9096   (1, 4)
 *   1295.1351   (1, 7)
 *   1629.7874   (1, 8)
 *
 * Note that the fundamental of a cylindrical shell is NOT (1, 1):
 * for a thin shell with h/R << 1, the bending term scales as
 * k(n^2+lambda^2)^4 and is minimised at moderate-to-high n. The
 * spectrum is also dense — modes (1, 4) through (1, 8) all sit
 * inside one octave, which is characteristic of cylindrical shells
 * and one reason their timbre is "metallic but unpitched".
 */

#include <chladni/analytical/shell.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <vector>

namespace {

constexpr double kTwoPi      = 2.0 * std::numbers::pi_v<double>;
// Reference values are quoted to 4 decimals (~1e-7 relative), so the
// tolerance only needs to absorb that quantization. A loose 1% gate
// previously masked a mis-transcribed cubic coefficient (the K1
// thinness term) for months — keep this tight.
constexpr double kRelTolerance = 1e-5;

}  // namespace

TEST_CASE("Steel SD-SD cylindrical shell R=0.1 L=0.2 h=1mm — first 5 frequencies",
          "[analytical][shell][cylindrical][donnell_mushtari]")
{
    const chladni::analytical::CylindricalShell geom{
        .radius    = 0.10,
        .length    = 0.20,
        .thickness = 1.0e-3,
    };
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0,
    };

    const std::vector<double> expected_hz = {
         962.2805,   //  (1, 5)
        1052.0696,   //  (1, 6)
        1124.9096,   //  (1, 4)
        1295.1351,   //  (1, 7)
        1629.7874,   //  (1, 8)
    };

    const auto omegas = chladni::analytical::
        simply_supported_cylindrical_shell_donnell_mushtari_angular_frequencies(
            geom, steel, expected_hz.size());

    REQUIRE(omegas.size() == expected_hz.size());

    // Frequencies must be returned in ascending order.
    for (std::size_t i = 1; i < omegas.size(); ++i) {
        INFO("ascending check at i=" << i
             << ": omegas[i-1]=" << omegas[i - 1]
             << "  omegas[i]="   << omegas[i]);
        REQUIRE(omegas[i] >= omegas[i - 1]);
    }

    for (std::size_t i = 0; i < expected_hz.size(); ++i) {
        const double f_hz   = omegas[i] / kTwoPi;
        const double rel_err = std::abs(f_hz - expected_hz[i]) / expected_hz[i];
        INFO("mode " << i
             << "  expected " << expected_hz[i] << " Hz"
             << "  got "      << f_hz          << " Hz"
             << "  rel_err "  << rel_err);
        REQUIRE(rel_err < kRelTolerance);
    }
}

TEST_CASE("Steel free-free cylindrical shell — Love inextensional formula",
          "[analytical][shell][cylindrical][free_free][rayleigh][love]")
{
    // Long thin steel cylinder where Love's inextensional approximation
    // is expected to be accurate (L/R = 10, h/R = 0.01). Reference values
    // are computed by direct evaluation of Leissa eq. (2.132) in NumPy.
    //
    //   D = E h^3 / (12 (1 - nu^2))
    //     = 200e9 * 1e-9 / (12 * 0.91)
    //     = 18.3150 N*m
    //   prefactor = D / (rho * h * R^4)
    //             = 18.3150 / (7850 * 1e-3 * 1e-4)
    //             = 23331.2 1/s^2
    //
    // Love correction with R^2/L^2 = 0.01:
    //   n=2: 1 + (24*0.7*0.01)/4 over 1 + (12*0.01)/(4*5)
    //      = (1 + 0.042) / (1 + 0.006)
    //      = 1.042 / 1.006 = 1.03579
    //   At L/R = 10 the correction moves the Rayleigh values by under
    //   2% for n=2..5, which is why this is the "long shell" regime.
    //
    //   omega_n^2 = prefactor * n^2 (n^2 - 1)^2 / (n^2 + 1) * love_factor
    const chladni::analytical::CylindricalShell geom{
        .radius    = 0.10,
        .length    = 1.00,
        .thickness = 1.0e-3,
    };
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0,
    };

    // Reference values are quoted to 4 decimals (~1e-6 relative); the
    // 1e-5 gate only absorbs that quantization. A loose 1% gate
    // previously masked a spurious (n^2-1) divisor in the Love
    // numerator (worth 1.37% at n=2 here) — keep this tight.
    //
    //   Rayleigh factor n^2 (n^2-1)^2 / (n^2+1) gives
    //     n=2: 7.2,    n=3: 57.6,  n=4: 211.76,  n=5: 553.85.
    const std::vector<double> expected_hz = {
         66.3881,   // n = 2
        186.0917,   // n = 3
        355.5397,   // n = 4
        573.9814,   // n = 5
    };

    const auto omegas = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom, steel, expected_hz.size());

    REQUIRE(omegas.size() == expected_hz.size());

    // Frequencies must be ascending in n for thin-shell inextensional.
    for (std::size_t i = 1; i < omegas.size(); ++i) {
        REQUIRE(omegas[i] >= omegas[i - 1]);
    }

    for (std::size_t i = 0; i < expected_hz.size(); ++i) {
        const double f_hz    = omegas[i] / kTwoPi;
        const double rel_err = std::abs(f_hz - expected_hz[i]) / expected_hz[i];
        INFO("mode " << i << " (n=" << (i + 2) << ")"
             << "  expected " << expected_hz[i] << " Hz"
             << "  got "      << f_hz          << " Hz"
             << "  rel_err "  << rel_err);
        REQUIRE(rel_err < 1.0e-5);
    }
}

TEST_CASE("Free-free Love correction approaches Rayleigh as L -> infinity",
          "[analytical][shell][cylindrical][free_free][rayleigh][love]")
{
    // For a fixed cross-section, increasing L should drive the Love
    // factor to 1 and the frequency toward Rayleigh's eq. (2.130).
    // Equivalently: doubling L from "long" to "very long" should
    // change the frequency by less than the residual Love correction
    // at the larger length.
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0,
    };
    const chladni::analytical::CylindricalShell geom_long{
        .radius = 0.10, .length = 10.0,    .thickness = 1.0e-3
    };
    const chladni::analytical::CylindricalShell geom_xlong{
        .radius = 0.10, .length = 1000.0,  .thickness = 1.0e-3
    };

    const auto w_long  = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom_long, steel, 4);
    const auto w_xlong = chladni::analytical::
        free_free_cylindrical_shell_inextensional_angular_frequencies(
            geom_xlong, steel, 4);

    // At L=1000 m the Love correction is below 1e-8, so w_xlong is
    // effectively Rayleigh. The first-order residual at L=10 m
    // (R^2/L^2 = 1e-4) is
    //   (24(1-nu)/n^2 - 12/(n^2(n^2+1)))/2 * R^2/L^2 ~= 1.8e-4
    // at n=2, decreasing with n — gate at 2.5e-4.
    for (std::size_t i = 0; i < w_long.size(); ++i) {
        const double rel = std::abs(w_long[i] - w_xlong[i]) / w_xlong[i];
        INFO("n=" << (i + 2) << " L=10:  " << w_long[i]
             << " rad/s  L=1000: " << w_xlong[i]
             << " rad/s  rel_diff=" << rel);
        REQUIRE(rel < 2.5e-4);
    }
}
