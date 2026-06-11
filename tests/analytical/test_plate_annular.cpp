/**
 * @file test_plate_annular.cpp
 * @brief Pin
 *   @ref chladni::analytical::annular_plate_clamped_clamped_angular_frequencies
 *   to the Vogel-Skinner / Leissa tabulation.
 *
 * The dimensionless frequency parameter
 * @f$ \lambda^{2} = \omega\,a^{2}\sqrt{\rho h / D} @f$ is independent of
 * material when @f$\nu@f$ is fixed. Leissa NASA SP-160 Table 2.18 gives
 * @f$\lambda^{2}@f$ at @f$\nu = 1/3@f$ for a clamped-clamped annulus
 * across several @f$b/a@f$ ratios. The lowest four modes at
 * @f$b/a = 0.5@f$ are
 *   (n=0, s=0): 89.2,
 *   (n=1, s=0): 90.2,
 *   (n=2, s=0): 93.3,
 *   (n=3, s=0): 99.0.
 * Modes with @f$n \ge 1@f$ are doubly degenerate (cos and sin in
 * @f$\theta@f$), so the implementation returns each twice and the
 * unique-frequency list above maps to the multiset
 *   {89.2, 90.2, 90.2, 93.3, 93.3, 99.0, 99.0}.
 *
 * Tolerance is 1.5%: Leissa quotes three significant figures and our
 * Bessel functions are accurate to ~5e-6 relative — the AS-1964
 * approximation noise dominates Leissa's rounding noise, so 1.5% is
 * comfortable.
 *
 * Reference: @cite leissa_1969_plates Table 2.18 (originally
 * @cite vogel_skinner_1965_annular).
 */

#include <chladni/analytical/plate.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr double kRelTolerance = 1.5e-2;

/// Convert Leissa's frequency parameter @f$\lambda^{2}@f$ to angular
/// frequency for the test geometry. omega = (lambda^2 / a^2) * sqrt(D / (rho h)).
double omega_from_lambda_squared(double lambda_squared,
                                 const chladni::analytical::AnnularPlate& geom,
                                 const chladni::IsotropicMaterial& mat)
{
    const double E   = mat.youngs_modulus;
    const double nu  = mat.poisson_ratio;
    const double rho = mat.density;
    const double a   = geom.radius_outer;
    const double h   = geom.thickness;
    const double D   = E * h * h * h / (12.0 * (1.0 - nu * nu));
    return (lambda_squared / (a * a)) * std::sqrt(D / (rho * h));
}

}  // namespace

TEST_CASE("Clamped-clamped annulus, b/a = 0.5, nu = 1/3 — first four modes",
          "[analytical][plate][annular]")
{
    using Catch::Matchers::WithinRel;

    const chladni::analytical::AnnularPlate geom{
        .radius_outer = 0.10,    // 10 cm
        .radius_inner = 0.05,    // 5 cm  →  b/a = 0.5
        .thickness    = 1.0e-3,  // 1 mm
    };
    // ν = 1/3 to match Leissa Table 2.18.
    const chladni::IsotropicMaterial mat{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0,
    };

    // Leissa Table 2.18 lambda^2 values at b/a = 0.5, ν = 1/3.
    // n=0 is non-degenerate; n>=1 is doubly degenerate, so the multiset
    // expands as 89.2, 90.2, 90.2, 93.3, 93.3, 99.0, 99.0.
    const std::vector<double> expected_lambda_squared = {
        89.2,  // (n=0, s=0)
        90.2,  // (n=1, s=0)  -- first copy
        90.2,  // (n=1, s=0)  -- degenerate pair
        93.3,  // (n=2, s=0)
        93.3,
        99.0,  // (n=3, s=0)
        99.0,
    };

    std::vector<double> expected_omegas;
    expected_omegas.reserve(expected_lambda_squared.size());
    for (double l2 : expected_lambda_squared) {
        expected_omegas.push_back(omega_from_lambda_squared(l2, geom, mat));
    }

    const auto omegas =
        chladni::analytical::annular_plate_clamped_clamped_angular_frequencies(
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
