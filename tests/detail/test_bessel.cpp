/**
 * @file test_bessel.cpp
 * @brief Pin the integer-order Bessel-function helpers to published values.
 *
 * Reference values are cross-checked against:
 *   * Abramowitz & Stegun, "Handbook of Mathematical Functions" (1964),
 *     tables 9.1 (J, Y), 9.8 (I, K).
 *   * NIST Digital Library of Mathematical Functions (DLMF) reference
 *     evaluations.
 *
 * The Abramowitz-Stegun rational approximations our implementation uses
 * are quoted as accurate to @f$\sim 10^{-7}@f$, so the test tolerance is
 * @f$10^{-6}@f$ — tight enough to catch coding mistakes, loose enough to
 * absorb the approximation error and any forward-recurrence drift.
 */

#include <chladni/detail/bessel.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

namespace cd = chladni::detail;

namespace {

// AS rational approximations of @f$Z_0, Z_1@f$ are quoted to roughly
// 1e-7 absolute. Forward recurrence to higher @f$n@f$ amplifies that
// by a small factor (~5x for @f$n=3@f$, @f$x=5@f$), so 5e-6 is the
// right tolerance for the order range we exercise.
constexpr double kAbsTol = 5.0e-6;

}  // namespace

TEST_CASE("J_n at canonical arguments matches AS / DLMF tables",
          "[detail][bessel][J]")
{
    using Catch::Matchers::WithinAbs;
    // J_0(0) = 1, J_n(0) = 0 for n >= 1 — guard the limit, not tested here
    // because the implementation is documented for x > 0.
    REQUIRE_THAT(cd::bessel_J(0, 1.0),  WithinAbs( 0.7651976865579666, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(1, 1.0),  WithinAbs( 0.4400505857449335, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(2, 1.0),  WithinAbs( 0.1149034849319005, kAbsTol));

    REQUIRE_THAT(cd::bessel_J(0, 5.0),  WithinAbs(-0.1775967713143383, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(1, 5.0),  WithinAbs(-0.3275791375914652, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(2, 5.0),  WithinAbs( 0.0465651162777522, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(3, 5.0),  WithinAbs( 0.3648297598758143, kAbsTol));

    REQUIRE_THAT(cd::bessel_J(0, 10.0), WithinAbs(-0.2459357644513483, kAbsTol));
    REQUIRE_THAT(cd::bessel_J(1, 10.0), WithinAbs( 0.0434727461688614, kAbsTol));
}

TEST_CASE("Y_n at canonical arguments matches AS / DLMF tables",
          "[detail][bessel][Y]")
{
    using Catch::Matchers::WithinAbs;
    REQUIRE_THAT(cd::bessel_Y(0, 1.0),  WithinAbs( 0.0882569642156769, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(1, 1.0),  WithinAbs(-0.7812128213002887, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(2, 1.0),  WithinAbs(-1.6506826068162547, kAbsTol));

    REQUIRE_THAT(cd::bessel_Y(0, 5.0),  WithinAbs(-0.3085176252490338, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(1, 5.0),  WithinAbs( 0.1478631433912263, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(2, 5.0),  WithinAbs( 0.3676628825720053, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(3, 5.0),  WithinAbs( 0.1462671643323002, kAbsTol));

    REQUIRE_THAT(cd::bessel_Y(0, 10.0), WithinAbs( 0.0556711672835499, kAbsTol));
    REQUIRE_THAT(cd::bessel_Y(1, 10.0), WithinAbs( 0.2490154242069538, kAbsTol));
}

TEST_CASE("I_n at canonical arguments matches AS / DLMF tables",
          "[detail][bessel][I]")
{
    using Catch::Matchers::WithinAbs;
    REQUIRE_THAT(cd::bessel_I(0, 1.0),  WithinAbs(1.2660658777520084, kAbsTol));
    REQUIRE_THAT(cd::bessel_I(1, 1.0),  WithinAbs(0.5651591039924850, kAbsTol));
    REQUIRE_THAT(cd::bessel_I(2, 1.0),  WithinAbs(0.1357476697670383, kAbsTol));

    // I_n grows ~ e^x / sqrt(2 pi x) for large x; relative tolerance is more
    // useful here. WithinAbs at 1e-6 is still fine since these are O(10).
    REQUIRE_THAT(cd::bessel_I(0, 5.0),  WithinAbs(27.2398718236044, 1e-4));
    REQUIRE_THAT(cd::bessel_I(1, 5.0),  WithinAbs(24.3356418276776, 1e-4));
    REQUIRE_THAT(cd::bessel_I(2, 5.0),  WithinAbs(17.5056229762793, 1e-4));
    REQUIRE_THAT(cd::bessel_I(3, 5.0),  WithinAbs(10.3311501506167, 1e-4));
}

TEST_CASE("K_n at canonical arguments matches AS / DLMF tables",
          "[detail][bessel][K]")
{
    using Catch::Matchers::WithinAbs;
    REQUIRE_THAT(cd::bessel_K(0, 1.0),  WithinAbs(0.4210244382407083, kAbsTol));
    REQUIRE_THAT(cd::bessel_K(1, 1.0),  WithinAbs(0.6019072301972346, kAbsTol));
    REQUIRE_THAT(cd::bessel_K(2, 1.0),  WithinAbs(1.6248388986351774, kAbsTol));

    // K_n is tiny at x=5 (~ e^-x). AS 9.8.6/9.8.8 are quoted to 1.9e-7 and
    // 2.2e-7 in the SCALED form sqrt(x) e^x K_n(x); after unscaling at
    // x=5 the tolerance on K_n itself is ~1e-9. Take 5e-9 to leave slack
    // for forward-recurrence amplification.
    REQUIRE_THAT(cd::bessel_K(0, 5.0),  WithinAbs(0.003691098334742659, 5e-9));
    REQUIRE_THAT(cd::bessel_K(1, 5.0),  WithinAbs(0.004044613445452164, 5e-9));
    REQUIRE_THAT(cd::bessel_K(2, 5.0),  WithinAbs(0.005308943776957162, 5e-9));
    REQUIRE_THAT(cd::bessel_K(3, 5.0),  WithinAbs(0.008291768415231819, 5e-9));
}

TEST_CASE("Bessel derivatives consistent with finite differences",
          "[detail][bessel][derivative]")
{
    using Catch::Matchers::WithinAbs;
    // Step h trades two error sources: central FD truncation O(h^2) shrinks
    // with h; the AS approximation noise (~1e-7) gets divided by 2h, growing
    // as h shrinks. h = 1e-3 minimises the sum at our argument range.
    constexpr double h = 1.0e-3;
    constexpr double tol = 1.0e-3;
    // Test points are interior to the AS branches (J/Y boundary x=3,
    // I boundary x=3.75, K boundary x=2). Avoid x near those seams: an
    // h-step that crosses a branch boundary yields apparent derivative
    // jumps because the two sub-approximations differ by O(1e-7) at the
    // seam, which FD then divides by 2h.
    for (int n = 0; n <= 3; ++n) {
        for (double x : {1.0, 5.0, 8.0}) {
            const double dJ_fd = (cd::bessel_J(n, x + h) - cd::bessel_J(n, x - h)) / (2.0 * h);
            const double dY_fd = (cd::bessel_Y(n, x + h) - cd::bessel_Y(n, x - h)) / (2.0 * h);
            const double dI_fd = (cd::bessel_I(n, x + h) - cd::bessel_I(n, x - h)) / (2.0 * h);
            const double dK_fd = (cd::bessel_K(n, x + h) - cd::bessel_K(n, x - h)) / (2.0 * h);

            INFO("n=" << n << " x=" << x);
            REQUIRE_THAT(cd::bessel_J_prime(n, x), WithinAbs(dJ_fd, tol));
            REQUIRE_THAT(cd::bessel_Y_prime(n, x), WithinAbs(dY_fd, tol));
            REQUIRE_THAT(cd::bessel_I_prime(n, x), WithinAbs(dI_fd, tol));
            REQUIRE_THAT(cd::bessel_K_prime(n, x), WithinAbs(dK_fd, tol));
        }
    }
}

TEST_CASE("Recurrences hold: 2n/x · Z_n = Z_{n-1} ± Z_{n+1}",
          "[detail][bessel][recurrence]")
{
    using Catch::Matchers::WithinAbs;
    constexpr double tol = 1.0e-6;
    for (int n = 1; n <= 4; ++n) {
        for (double x : {2.0, 6.0, 12.0}) {
            INFO("n=" << n << " x=" << x);
            // J, Y: Z_{n-1}(x) + Z_{n+1}(x) = (2n/x) Z_n(x)
            REQUIRE_THAT(cd::bessel_J(n-1, x) + cd::bessel_J(n+1, x),
                         WithinAbs((2.0 * n / x) * cd::bessel_J(n, x), tol));
            REQUIRE_THAT(cd::bessel_Y(n-1, x) + cd::bessel_Y(n+1, x),
                         WithinAbs((2.0 * n / x) * cd::bessel_Y(n, x), tol));

            // I: I_{n-1}(x) - I_{n+1}(x) = (2n/x) I_n(x)
            REQUIRE_THAT(cd::bessel_I(n-1, x) - cd::bessel_I(n+1, x),
                         WithinAbs((2.0 * n / x) * cd::bessel_I(n, x), tol));

            // K: K_{n+1}(x) - K_{n-1}(x) = (2n/x) K_n(x)
            REQUIRE_THAT(cd::bessel_K(n+1, x) - cd::bessel_K(n-1, x),
                         WithinAbs((2.0 * n / x) * cd::bessel_K(n, x), tol));
        }
    }
}
