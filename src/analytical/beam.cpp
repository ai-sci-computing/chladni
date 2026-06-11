/**
 * @file beam.cpp
 * @brief Implementation of the analytical Euler-Bernoulli beam references
 *        declared in @ref include/chladni/analytical/beam.hpp.
 *
 * The free-free eigenvalue equation
 * @f$\cos(x)\cosh(x) = 1@f$ has roots
 * @f$x_n = \beta_n L \approx 4.7300, 7.8532, 10.9956, \ldots@f$,
 * with the n-th root close to @f$(n + 1/2)\pi@f$ for large @f$n@f$.
 * We bracket each root in @f$[\,n\pi, (n+1)\pi\,]@f$ — across that
 * interval @f$\cos@f$ traverses a full period and the @f$\cosh@f$
 * factor is monotone, so the function takes opposite signs at the
 * endpoints — and bisect.
 */

#include <chladni/analytical/beam.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace chladni::analytical {

namespace {

constexpr double kPi = std::numbers::pi_v<double>;

/// Free-free transcendental equation, expressed as the function whose
/// zero we seek. Written so that the amplitude stays in a comfortable
/// numerical range even at large x: factor out cosh(x) before evaluating
/// the characteristic combination.
double free_free_residual(double x)
{
    // cos(x) cosh(x) - 1. cosh grows exponentially; for x = 17 (5th
    // root) cosh(x) ~ 1.2e7, but the product cos(x) cosh(x) - 1 is
    // still well within double range, so write it directly.
    return std::cos(x) * std::cosh(x) - 1.0;
}

double bisect_to_zero(double lo, double hi, double f_lo,
                      int max_iter = 80, double tol = 1.0e-13)
{
    // Caller guarantees f_lo and f(hi) have opposite sign and both finite.
    for (int it = 0; it < max_iter; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = free_free_residual(mid);
        if (std::abs(f_mid) < tol
            || (hi - lo) < tol * std::max(1.0, std::abs(mid))) {
            return mid;
        }
        if ((f_lo < 0.0) == (f_mid < 0.0)) {
            lo   = mid;
            f_lo = f_mid;
        } else {
            hi   = mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // anonymous namespace

std::vector<double> free_free_beam_eigenvalue_roots(std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "free_free_beam_eigenvalue_roots: n_modes must be >= 1");
    }
    std::vector<double> roots;
    roots.reserve(n_modes);

    // The n-th positive root of cos(x) cosh(x) = 1 lies in the interval
    // (n pi, (n+1) pi). Verify the sign change and bisect.
    for (std::size_t n = 1; roots.size() < n_modes; ++n) {
        const double lo = static_cast<double>(n) * kPi;
        const double hi = static_cast<double>(n + 1) * kPi;
        const double f_lo = free_free_residual(lo);
        const double f_hi = free_free_residual(hi);
        // The signs alternate: f(n*pi) = cos(n*pi) cosh(n*pi) - 1 =
        // (-1)^n cosh(n*pi) - 1, large in magnitude with sign (-1)^n
        // for n >= 1. So f_lo * f_hi < 0 always.
        if (f_lo * f_hi >= 0.0) {
            // Should not happen for any n >= 1; surface as runtime_error.
            throw std::runtime_error(
                "free_free_beam_eigenvalue_roots: bracket failed to "
                "produce a sign change; this indicates a bug");
        }
        roots.push_back(bisect_to_zero(lo, hi, f_lo));
    }
    return roots;
}

std::vector<double> free_free_beam_angular_frequencies(
    const RectangularBeam& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "free_free_beam_angular_frequencies: n_modes must be >= 1");
    }
    if (geom.length <= 0.0 || geom.width <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "free_free_beam_angular_frequencies: all geometric "
            "dimensions must be > 0");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "free_free_beam_angular_frequencies: Young's modulus and "
            "density must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "free_free_beam_angular_frequencies: Poisson's ratio must "
            "lie in (-1, 1/2)");
    }

    const double E = material.youngs_modulus;
    const double rho = material.density;
    const double L = geom.length;
    const double w = geom.width;
    const double h = geom.thickness;
    const double A = w * h;
    const double I = w * h * h * h / 12.0;

    // omega_n^2 = (beta_n L)^4 * E I / (rho A L^4); pre-factor is a fixed
    // scale independent of mode.
    const double scale = std::sqrt(E * I / (rho * A * std::pow(L, 4.0)));

    const auto roots = free_free_beam_eigenvalue_roots(n_modes);
    std::vector<double> omegas;
    omegas.reserve(n_modes);
    for (double bL : roots) {
        omegas.push_back((bL * bL) * scale);
    }
    // Already ascending because bL is ascending.
    return omegas;
}

}  // namespace chladni::analytical
