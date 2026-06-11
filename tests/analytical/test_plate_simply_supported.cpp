/**
 * @file test_plate_simply_supported.cpp
 * @brief Phase-1 gating test: simply-supported rectangular thin plate.
 *
 * Pins
 * @ref chladni::analytical::simply_supported_rect_plate_angular_frequencies
 * to hand-computed reference values for a thin aluminum plate
 * (@f$E = 69\,\text{GPa}@f$, @f$\nu = 0.33@f$,
 *  @f$\rho = 2700\,\text{kg/m}^3@f$, @f$h = 1\,\text{mm}@f$,
 *  @f$a = 0.5\,\text{m}@f$, @f$b = 0.3\,\text{m}@f$).
 * Reference values were obtained by direct evaluation of the closed-form
 * Leissa (1969, NASA SP-160) eq. for a Navier-supported rectangle.
 *
 * Expected ordered Hz spectrum, first six modes:
 *   (1,1)   36.70    (2,1)   65.84    (3,1)  114.4
 *   (1,2)  117.7     (2,2)  146.8     (4,1)  182.4
 *
 * @note (4,1) at bracket = 16/a^2 + 1/b^2 = 75.11 m^-2 falls between (2,2)
 *       at 60.44 m^-2 and (1,3) at 104.0 m^-2 because the plate is
 *       elongated along x (a > b), so high-x modes are cheaper.
 */

#include <chladni/analytical/plate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <vector>

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;

// 1% relative tolerance: tight enough to catch implementation mistakes,
// loose enough to absorb 4-significant-figure rounding in the expected
// values below.
constexpr double kRelTolerance = 1e-2;

}  // namespace

TEST_CASE("Aluminum 0.5m x 0.3m x 1mm plate — first six natural frequencies",
          "[analytical][plate][simply_supported]")
{
    const chladni::analytical::RectanglePlate geom{
        .length_a  = 0.50,
        .length_b  = 0.30,
        .thickness = 1.0e-3,
    };
    const chladni::IsotropicMaterial aluminum{
        .youngs_modulus = 69.0e9,
        .poisson_ratio  = 0.33,
        .density        = 2700.0,
    };

    // Hand-computed expected frequencies in Hz, sorted ascending.
    const std::vector<double> expected_hz = {
         36.70,  //  (1,1)
         65.84,  //  (2,1)
        114.4,   //  (3,1)
        117.7,   //  (1,2)
        146.8,   //  (2,2)
        182.4,   //  (4,1)  -- a > b, so (4,1) precedes (1,3)
    };

    const auto omegas =
        chladni::analytical::simply_supported_rect_plate_angular_frequencies(
            geom, aluminum, expected_hz.size());

    REQUIRE(omegas.size() == expected_hz.size());

    // Frequencies must be returned in ascending order.
    for (std::size_t i = 1; i < omegas.size(); ++i) {
        INFO("ascending order check at i=" << i
             << ": omegas[i-1]=" << omegas[i - 1]
             << " omegas[i]="   << omegas[i]);
        REQUIRE(omegas[i] >= omegas[i - 1]);
    }

    for (std::size_t i = 0; i < expected_hz.size(); ++i) {
        const double f_hz   = omegas[i] / kTwoPi;
        const double rel_err = std::abs(f_hz - expected_hz[i]) / expected_hz[i];
        INFO("mode " << i
             << " expected " << expected_hz[i] << " Hz"
             << "  got "      << f_hz          << " Hz"
             << "  rel_err "  << rel_err);
        REQUIRE(rel_err < kRelTolerance);
    }
}
