/**
 * @file test_beam_free_free.cpp
 * @brief Pin
 *   @ref chladni::analytical::free_free_beam_eigenvalue_roots
 *   and
 *   @ref chladni::analytical::free_free_beam_angular_frequencies
 *   to published reference values.
 *
 * The five lowest @f$\beta_n L@f$ roots of @f$\cos x \cosh x = 1@f$
 * are universal constants tabulated in any vibrations textbook
 * (Inman *Engineering Vibration* 4th ed., Rao *Mechanical Vibrations*
 * ch. 8, Rayleigh *Theory of Sound* I §170). The asymptotic estimate
 * @f$\beta_n L \to (n + 1/2)\pi@f$ is excellent already at @f$n = 2@f$.
 *
 * The dimensional-frequency test uses a 1 m × 10 mm × 10 mm steel bar.
 * Both the eigenvalue equation and the universal-roots-to-frequency
 * conversion are exact in Euler-Bernoulli theory, so the only error
 * sources are bisection convergence and (negligible) floating-point
 * roundoff. Tolerance is therefore tight (@f$10^{-9}@f$ relative).
 */

#include <chladni/analytical/beam.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr double kRootAbsTol = 1.0e-9;
constexpr double kFreqRelTol = 1.0e-9;

}  // namespace

TEST_CASE("Universal beta_n L roots match the published tabulation",
          "[analytical][beam][free_free]")
{
    using Catch::Matchers::WithinAbs;
    // Reference: any standard vibrations text; values to 7 decimals are
    // those quoted in e.g. Rao Mechanical Vibrations Table 8.4. The
    // asymptotic limit is (n + 1/2) pi.
    const std::vector<double> expected = {
         4.730040744862704,
         7.853204624095838,
        10.995607838001671,
        14.137165491257464,
        17.278759657399481,
    };

    const auto roots =
        chladni::analytical::free_free_beam_eigenvalue_roots(expected.size());

    REQUIRE(roots.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        INFO("root " << i
             << " expected " << expected[i]
             << " got "      << roots[i]);
        REQUIRE_THAT(roots[i], WithinAbs(expected[i], kRootAbsTol));
    }
}

TEST_CASE("1 m × 10 mm × 10 mm steel free-free beam — first five frequencies",
          "[analytical][beam][free_free]")
{
    using Catch::Matchers::WithinRel;

    // 1 m steel bar, square 10 mm cross-section.
    const chladni::analytical::RectangularBeam geom{
        .length    = 1.00,
        .width     = 1.0e-2,
        .thickness = 1.0e-2,
    };
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0,
    };

    // Construct the expected omegas directly from the universal roots,
    // omega_n^2 = (beta_n L)^4 E I / (rho A L^4). This is just exercising
    // that the dimensional helper is consistent with the universal-root
    // helper for a known geometry. With A = w h and I = w h^3 / 12 and
    // L = 1 m the prefactor sqrt(E I / (rho A L^4)) is fixed.
    const double E = steel.youngs_modulus;
    const double rho = steel.density;
    const double w = geom.width;
    const double h = geom.thickness;
    const double L = geom.length;
    const double A = w * h;
    const double I = w * h * h * h / 12.0;
    const double scale = std::sqrt(E * I / (rho * A * std::pow(L, 4)));

    const auto roots =
        chladni::analytical::free_free_beam_eigenvalue_roots(5);
    std::vector<double> expected_omegas;
    expected_omegas.reserve(roots.size());
    for (double bL : roots) {
        expected_omegas.push_back((bL * bL) * scale);
    }

    const auto omegas =
        chladni::analytical::free_free_beam_angular_frequencies(
            geom, steel, expected_omegas.size());

    REQUIRE(omegas.size() == expected_omegas.size());

    for (std::size_t i = 0; i < expected_omegas.size(); ++i) {
        INFO("mode " << i
             << " expected omega = " << expected_omegas[i]
             << "  got "             << omegas[i]);
        REQUIRE_THAT(omegas[i], WithinRel(expected_omegas[i], kFreqRelTol));
    }

    // Also pin the fundamental in absolute terms, so the test has at least
    // one externally-anchored number that can't be self-consistent.
    // omega_1 = (4.7300407...)^2 sqrt(EI / (rho A L^4))
    //        = 22.3733  * sqrt(2e11 * 8.3333e-10 / (7850 * 1e-4 * 1))
    //        = 22.3733  * sqrt(166.6667 / 0.785)
    //        = 22.3733  * sqrt(212.314)  =  22.3733 * 14.5710
    //        ~ 325.94 rad/s   (~ 51.87 Hz)
    REQUIRE_THAT(omegas[0], WithinRel(325.939, 1e-3));
}
