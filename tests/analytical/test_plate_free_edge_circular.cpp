/**
 * @file test_plate_free_edge_circular.cpp
 * @brief Pin
 *   @ref chladni::analytical::free_edge_circular_plate_angular_frequencies
 *   to Leissa's tabulation. This is the eigenvalue family of Chladni's
 *   classical sand-pattern experiments.
 *
 * Reference values from @cite leissa_1969_plates Table 2.5 at
 * @f$\nu = 0.33@f$. The lowest eight rows of that table give
 * @f$\lambda^{2}@f$ for @f$(n, s)@f$:
 *
 *   (n=2, s=0):  5.253     (n=4, s=0): 21.6  (asterisk: ~2% accurate)
 *   (n=0, s=1):  9.084     (n=2, s=1): 35.25
 *   (n=3, s=0): 12.23      (n=5, s=0): 33.1  (asterisk)
 *   (n=1, s=1): 20.52      (n=0, s=2): 38.55
 *
 * After expanding @f$n \ge 1@f$ modes with their cos/sin degenerate
 * pair (and treating @f$n = 0@f$ as non-degenerate), the lowest seven
 * angular frequencies correspond to
 *   {5.253, 5.253, 9.084, 12.23, 12.23, 20.52, 20.52}.
 * (Stopping at 7 because the 8th would be the asterisk-marked 21.6
 * from the approximate eq. 2.16, which Leissa flags as ~2% accurate
 * — too loose for a test pin.)
 *
 * Tolerance is 0.5%: Leissa quotes 4 sig figs; the Bessel
 * approximations are good to ~5e-6 relative; the bisection refines
 * each root to ~1e-10 relative. So the dominant uncertainty is
 * Leissa's rounding (≤ 0.05% on each value), and 0.5% leaves
 * comfortable headroom.
 */

#include <chladni/analytical/plate.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr double kRelTolerance = 5.0e-3;

double omega_from_lambda_squared(double lambda_squared,
                                 const chladni::analytical::CircularPlate& geom,
                                 const chladni::IsotropicMaterial& mat)
{
    const double E   = mat.youngs_modulus;
    const double nu  = mat.poisson_ratio;
    const double rho = mat.density;
    const double a   = geom.radius;
    const double h   = geom.thickness;
    const double D   = E * h * h * h / (12.0 * (1.0 - nu * nu));
    return (lambda_squared / (a * a)) * std::sqrt(D / (rho * h));
}

}  // namespace

TEST_CASE("Free-edge circular plate, nu = 0.33 — lowest seven modes",
          "[analytical][plate][free_edge_circular]")
{
    using Catch::Matchers::WithinRel;

    const chladni::analytical::CircularPlate geom{
        .radius    = 0.10,    // 10 cm
        .thickness = 1.0e-3,  // 1 mm
    };
    // Leissa Table 2.5 is at nu = 0.33.
    const chladni::IsotropicMaterial mat{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.33,
        .density        = 7850.0,
    };

    const std::vector<double> expected_lambda_squared = {
         5.253,  // (n=2, s=0)
         5.253,  // (n=2, s=0)  -- degenerate
         9.084,  // (n=0, s=1)  -- non-degenerate
        12.23,   // (n=3, s=0)
        12.23,
        20.52,   // (n=1, s=1)
        20.52,
    };

    std::vector<double> expected_omegas;
    expected_omegas.reserve(expected_lambda_squared.size());
    for (double l2 : expected_lambda_squared) {
        expected_omegas.push_back(omega_from_lambda_squared(l2, geom, mat));
    }

    const auto omegas =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            geom, mat, expected_omegas.size());

    REQUIRE(omegas.size() == expected_omegas.size());

    // Frequencies must be returned in ascending order.
    for (std::size_t i = 1; i < omegas.size(); ++i) {
        INFO("ascending check at i=" << i
             << ": omegas[i-1]=" << omegas[i - 1]
             << " omegas[i]="    << omegas[i]);
        REQUIRE(omegas[i] >= omegas[i - 1]);
    }

    for (std::size_t i = 0; i < expected_omegas.size(); ++i) {
        INFO("mode " << i
             << " expected omega = " << expected_omegas[i]
             << "  got "             << omegas[i]
             << "  rel_err = "       << std::abs(omegas[i] - expected_omegas[i])
                                        / expected_omegas[i]);
        REQUIRE_THAT(omegas[i], WithinRel(expected_omegas[i], kRelTolerance));
    }
}
