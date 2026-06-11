/**
 * @file plate.cpp
 * @brief Implementation of the analytical plate-vibration references.
 *
 * Implements the Kirchhoff thin-plate closed forms documented in
 * @ref include/chladni/analytical/plate.hpp.
 * Reference: @cite leissa_1969_plates, ch. 4 (rectangular plates).
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/detail/bessel.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace chladni::analytical {

namespace {

/// Validates inputs to the simply-supported plate solver. Throws
/// std::invalid_argument on any non-physical or degenerate input.
void validate_inputs(const RectanglePlate& geom,
                     const ::chladni::IsotropicMaterial& material,
                     std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "simply_supported_rect_plate_angular_frequencies: "
            "n_modes must be >= 1");
    }
    if (geom.length_a <= 0.0 || geom.length_b <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "simply_supported_rect_plate_angular_frequencies: "
            "all geometric dimensions must be strictly positive");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "simply_supported_rect_plate_angular_frequencies: "
            "Young's modulus and density must be strictly positive");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "simply_supported_rect_plate_angular_frequencies: "
            "Poisson's ratio must lie in (-1, 1/2)");
    }
}

/// Upper bound on the per-axis mode index we have to enumerate to be sure
/// the smallest @p n_modes (m,n) frequencies are captured even for highly
/// anisotropic plates. Derivation: the n_modes-th smallest value of the
/// bracket m^2/a^2 + n^2/b^2 has both m^2/a^2 and n^2/b^2 bounded by it,
/// hence m and n are each bounded by sqrt(n_modes * max(a^2,b^2)/min(a^2,b^2)).
/// We add a small constant slack so the bound is strict, not tight.
std::size_t mode_index_bound(double a, double b, std::size_t n_modes) noexcept
{
    const double aspect2 = std::max(a * a / (b * b), b * b / (a * a));
    const double raw = std::sqrt(static_cast<double>(n_modes) * aspect2) + 1.0;
    const auto cast = static_cast<std::size_t>(std::ceil(raw));
    return std::max<std::size_t>(cast, 2);
}

/// Bisect a sign change of @p f on @p [lo, hi]. Caller guarantees
/// @c f(lo) and @c f(hi) have opposite sign and are both finite.
/// Used by both the free-edge circular plate solver and the
/// clamped-clamped annular plate solver to refine determinant zeros.
///
/// Returns NaN if the bracket turns out to straddle a determinant pole /
/// overflow rather than a zero (a finite opposite-sign pair can bracket a
/// singularity, e.g. I_n overflowing at large λ); callers must drop NaN
/// results instead of recording them as frequencies.
template <class F>
double bisect(F&& f, double lo, double hi, double f_lo,
              int max_iter = 80, double tol = 1.0e-10)
{
    for (int it = 0; it < max_iter; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = f(mid);
        if (!std::isfinite(f_mid)) {
            // Pole / overflow inside the bracket, not a root — reject so it is
            // not recorded as a spurious frequency.
            return std::numeric_limits<double>::quiet_NaN();
        }
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

std::vector<double> simply_supported_rect_plate_angular_frequencies(
    const RectanglePlate& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    validate_inputs(geom, material, n_modes);

    constexpr double pi = std::numbers::pi_v<double>;
    const double a   = geom.length_a;
    const double b   = geom.length_b;
    const double h   = geom.thickness;
    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;

    // Flexural rigidity D = E h^3 / (12 (1 - nu^2))   (Leissa eq. (1.4)).
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));

    // Angular-frequency coefficient K = pi^2 sqrt(D / (rho h)).
    const double K = pi * pi * std::sqrt(D / (rho * h));

    // Enumerate (m, n) candidates and collect their frequencies.
    const std::size_t M = mode_index_bound(a, b, n_modes);
    std::vector<double> omegas;
    omegas.reserve(M * M);
    for (std::size_t m = 1; m <= M; ++m) {
        for (std::size_t n = 1; n <= M; ++n) {
            const auto m_d = static_cast<double>(m);
            const auto n_d = static_cast<double>(n);
            const double bracket = (m_d * m_d) / (a * a) + (n_d * n_d) / (b * b);
            omegas.push_back(K * bracket);
        }
    }

    // Lowest n_modes entries form the spectrum we report. partial_sort puts
    // the smallest n_modes at the front (in sorted order) without sorting
    // the entire enumeration tail.
    std::partial_sort(omegas.begin(), omegas.begin() + static_cast<std::ptrdiff_t>(n_modes),
                      omegas.end());
    omegas.resize(n_modes);
    return omegas;
}

// ---------------------------------------------------------------------------
// Free-edge circular plate (Chladni's namesake configuration).
// ---------------------------------------------------------------------------

namespace {

/// Frequency-equation residual for a free-edge solid circular plate at
/// circumferential mode @p n, dimensionless argument @p lam = ka, and
/// Poisson ratio @p nu. The four building blocks M_J, M_I, V_J, V_I are
/// derived by substituting W = A J_n(kr) + C I_n(kr) into the Kirchhoff
/// free-edge boundary conditions M_r(a) = 0 and V_r(a) = 0; the
/// determinant of the resulting 2x2 system in (A, C) is M_J V_I - M_I V_J.
double free_edge_residual(int n, double lam, double nu)
{
    using namespace ::chladni::detail;
    const double Jn = bessel_J(n, lam);
    const double Jp = bessel_J_prime(n, lam);
    const double In = bessel_I(n, lam);
    const double Ip = bessel_I_prime(n, lam);
    const double n2 = static_cast<double>(n) * static_cast<double>(n);
    const double lam2 = lam * lam;
    const double lam3 = lam2 * lam;
    const double one_minus_nu = 1.0 - nu;

    const double M_J = lam2 * Jn + one_minus_nu * (lam * Jp - n2 * Jn);
    const double M_I = lam2 * In - one_minus_nu * (lam * Ip - n2 * In);
    const double V_J = lam3 * Jp + one_minus_nu * n2 * (lam * Jp - Jn);
    const double V_I = lam3 * Ip - one_minus_nu * n2 * (lam * Ip - In);

    return M_J * V_I - M_I * V_J;
}

void validate_circular_inputs(const CircularPlate& geom,
                              const ::chladni::IsotropicMaterial& material,
                              std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "free_edge_circular_plate_angular_frequencies: "
            "n_modes must be >= 1");
    }
    if (geom.radius <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "free_edge_circular_plate_angular_frequencies: "
            "radius and thickness must be > 0");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "free_edge_circular_plate_angular_frequencies: "
            "Young's modulus and density must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "free_edge_circular_plate_angular_frequencies: "
            "Poisson's ratio must lie in (-1, 1/2)");
    }
}

}  // anonymous namespace

std::vector<double> free_edge_circular_plate_angular_frequencies(
    const CircularPlate& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    validate_circular_inputs(geom, material, n_modes);

    const double a   = geom.radius;
    const double h   = geom.thickness;
    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;

    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));
    const double omega_per_lambda2 = (1.0 / (a * a)) * std::sqrt(D / (rho * h));

    // Search window in lambda. Start above 0 to skip the rigid-body
    // kernel (the residual is identically zero at lambda = 0 because
    // every term carries a positive power of lambda). The lowest
    // physical mode at each n sits roughly at lambda ≈ max(3, n) for
    // n >= 2 (Leissa Table 2.5 row s=0), and at lambda > 3 for n=0,1.
    constexpr double kLambdaMinFloor = 1.0;
    const double lambda_max =
        15.0 + 4.0 * std::sqrt(static_cast<double>(n_modes));
    constexpr int kSamplesPerUnitLambda = 50;
    const int total_samples =
        std::max(64, static_cast<int>(kSamplesPerUnitLambda * lambda_max));

    const int n_max = static_cast<int>(n_modes) + 4;

    struct Root { int n; double lambda; };
    std::vector<Root> roots;
    roots.reserve(static_cast<std::size_t>(n_max) * 4);

    for (int n = 0; n <= n_max; ++n) {
        auto f = [n, nu](double lam) {
            return free_edge_residual(n, lam, nu);
        };

        // Forward Bessel recurrence is well-conditioned for n <= x;
        // beyond that it loses ~3 sig figs per recurrence step. Start
        // each per-n scan at max(kLambdaMinFloor, double(n)) — for the
        // free-edge circular plate the lowest physical eigenvalue at
        // circumferential index n satisfies lambda > n - 1 already at
        // n = 2, and approaches lambda ≈ (n + 2s) pi / 2 for higher
        // (n, s) pairs (Leissa eq. 2.17), so the floor max(1, n) is
        // safe for the lowest few modes per n.
        const double lam_min_n = std::max(kLambdaMinFloor,
                                          static_cast<double>(n));
        if (lam_min_n >= lambda_max) {
            break;
        }
        const double dlam = (lambda_max - kLambdaMinFloor)
                          / static_cast<double>(total_samples);
        const int first_step =
            static_cast<int>(std::ceil((lam_min_n - kLambdaMinFloor) / dlam))
            + 1;

        double lam_prev = kLambdaMinFloor + dlam * (first_step - 1);
        double f_prev   = f(lam_prev);
        // Detects sign changes only; an even-multiplicity (tangent) root would
        // be missed, but the free-edge circular-plate radial determinant has
        // only simple roots in this window, so that case does not arise here.
        for (int s = first_step; s <= total_samples; ++s) {
            const double lam = kLambdaMinFloor + dlam * s;
            const double f_curr = f(lam);
            if (std::isfinite(f_prev) && std::isfinite(f_curr)
                && f_prev * f_curr < 0.0) {
                const double r = bisect(f, lam_prev, lam, f_prev);
                if (std::isfinite(r)) {
                    roots.push_back({n, r});  // drop poles flagged as NaN
                }
            }
            lam_prev = lam;
            f_prev   = f_curr;
        }
    }

    if (roots.empty()) {
        throw std::runtime_error(
            "free_edge_circular_plate_angular_frequencies: "
            "no determinant zeros found in the search window");
    }

    // n=0 modes are non-degenerate (axisymmetric), n>=1 modes are
    // doubly degenerate (cos/sin in theta).
    std::vector<double> omegas;
    omegas.reserve(roots.size() * 2);
    for (const auto& r : roots) {
        const double omega = r.lambda * r.lambda * omega_per_lambda2;
        omegas.push_back(omega);
        if (r.n >= 1) {
            omegas.push_back(omega);
        }
    }

    if (omegas.size() < n_modes) {
        throw std::runtime_error(
            "free_edge_circular_plate_angular_frequencies: "
            "search window did not capture enough modes");
    }

    std::partial_sort(
        omegas.begin(),
        omegas.begin() + static_cast<std::ptrdiff_t>(n_modes),
        omegas.end());
    omegas.resize(n_modes);
    return omegas;
}

// ---------------------------------------------------------------------------
// Annular plate, clamped on both edges.
// ---------------------------------------------------------------------------

namespace {

/// Validate the annular-plate inputs. Throws std::invalid_argument on any
/// degenerate or non-physical configuration.
void validate_annular_inputs(const AnnularPlate& geom,
                             const ::chladni::IsotropicMaterial& material,
                             std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "n_modes must be >= 1");
    }
    if (geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "thickness must be > 0");
    }
    if (geom.radius_outer <= 0.0 || geom.radius_inner <= 0.0
        || geom.radius_inner >= geom.radius_outer) {
        throw std::invalid_argument(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "must have 0 < radius_inner < radius_outer");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "Young's modulus and density must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "Poisson's ratio must lie in (-1, 1/2)");
    }
}

/// Frequency determinant for clamped-clamped annulus, circumferential
/// mode @p n, dimensionless frequency @p lam (= ka), aspect @p alpha (= b/a).
/// Returns the value of the 4x4 determinant; we only need its sign for
/// bisection.
double clamped_clamped_determinant(int n, double lam, double alpha)
{
    using namespace ::chladni::detail;
    const double a_lam = alpha * lam;
    Eigen::Matrix4d M;
    M(0, 0) = bessel_J(n, lam);          M(0, 1) = bessel_Y(n, lam);
    M(0, 2) = bessel_I(n, lam);          M(0, 3) = bessel_K(n, lam);
    M(1, 0) = bessel_J(n, a_lam);        M(1, 1) = bessel_Y(n, a_lam);
    M(1, 2) = bessel_I(n, a_lam);        M(1, 3) = bessel_K(n, a_lam);
    M(2, 0) = bessel_J_prime(n, lam);    M(2, 1) = bessel_Y_prime(n, lam);
    M(2, 2) = bessel_I_prime(n, lam);    M(2, 3) = bessel_K_prime(n, lam);
    M(3, 0) = bessel_J_prime(n, a_lam);  M(3, 1) = bessel_Y_prime(n, a_lam);
    M(3, 2) = bessel_I_prime(n, a_lam);  M(3, 3) = bessel_K_prime(n, a_lam);

    // The raw determinant of M is a smooth, continuous function of lambda;
    // its zeros are exactly the natural-frequency eigenvalues. We tried
    // column-scaling to a unit max for numerical conditioning, but that
    // makes the determinant DIScontinuous (the column max jumps from one
    // row to another as lambda varies, scaling the corresponding column
    // by a different factor on either side of the jump and producing
    // spurious sign changes). Eigen's 4x4 closed-form determinant handles
    // the dynamic range of I_n / K_n in our search window in stride.
    return M.determinant();
}

}  // anonymous namespace

std::vector<double> annular_plate_clamped_clamped_angular_frequencies(
    const AnnularPlate& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    validate_annular_inputs(geom, material, n_modes);

    const double a   = geom.radius_outer;
    const double b   = geom.radius_inner;
    const double h   = geom.thickness;
    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;

    const double alpha = b / a;
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));
    // omega = (lambda^2 / a^2) sqrt(D / (rho h)).
    const double omega_per_lambda2 = (1.0 / (a * a)) * std::sqrt(D / (rho * h));

    // Search window in lambda. Lower bound chosen well above the Y_n / K_n
    // logarithmic singularities at the origin: per @cite leissa_1969_plates
    // Table 2.17 the fundamental clamped-clamped annulus mode satisfies
    // lambda > 3 across all b/a ratios in (0, 1), so 2.0 is safe and avoids
    // the numerical noise that creates spurious sign changes for lambda < 1.
    // Upper bound chosen to give a generous superset of the requested
    // mode count; lambda^2 ~ ((s + 1) pi / (1 - alpha))^2 + n^2 for the
    // (n, s) mode is a useful order-of-magnitude check.
    constexpr double kLambdaMin = 2.0;
    const double lambda_max =
        20.0 + 4.0 * std::sqrt(static_cast<double>(n_modes));
    constexpr int kSamplesPerUnitLambda = 40;  // 0.025 step is plenty fine
    const int total_samples =
        std::max(64, static_cast<int>(kSamplesPerUnitLambda * lambda_max));

    // Highest circumferential wave number to scan. Modes (n, s=0) for
    // n = 0, 1, 2, ... fill the low spectrum, so we need at least
    // n_max >= n_modes when modes are non-degenerate, and a smaller value
    // suffices once degeneracy doubles each n>=1 contribution. Add slack.
    const int n_max =
        static_cast<int>(n_modes) + 4;

    struct Root { int n; double lambda; };
    std::vector<Root> roots;
    roots.reserve(static_cast<std::size_t>(n_max) * 4);

    for (int n = 0; n <= n_max; ++n) {
        auto f = [n, alpha](double lam) {
            return clamped_clamped_determinant(n, lam, alpha);
        };

        // Forward Bessel recurrence (used to build J_n / Y_n / I_n /
        // K_n for n >= 2 in the detail::bessel module) is well-
        // conditioned only when n <= x. For lam < n it loses accuracy
        // catastrophically, producing spurious sign changes in the
        // determinant. Any physical clamped-clamped annulus eigenvalue
        // for circumferential index n always sits at lam > n + 1 (cf.
        // @cite leissa_1969_plates Table 2.6 for the b/a -> 0 limit
        // and Table 2.18 for finite b/a), so starting each per-n scan
        // at max(kLambdaMin, n + 1) skips the unreliable region without
        // dropping any physical mode.
        const double lam_min_n = std::max(kLambdaMin,
                                          static_cast<double>(n) + 1.0);
        if (lam_min_n >= lambda_max) {
            break;
        }
        double lam_prev = lam_min_n;
        double f_prev   = f(lam_prev);
        const double dlam = (lambda_max - kLambdaMin)
                          / static_cast<double>(total_samples);
        const int first_step =
            static_cast<int>(std::ceil((lam_min_n - kLambdaMin) / dlam)) + 1;
        // Sign-change detection only (simple roots); see the note in the
        // free-edge solver. Poles bracketed by finite opposite-sign samples
        // come back from bisect as NaN and are dropped.
        for (int s = first_step; s <= total_samples; ++s) {
            const double lam = kLambdaMin + dlam * s;
            const double f_curr = f(lam);
            if (std::isfinite(f_prev) && std::isfinite(f_curr)
                && f_prev * f_curr < 0.0) {
                const double r = bisect(f, lam_prev, lam, f_prev);
                if (std::isfinite(r)) {
                    roots.push_back({n, r});
                }
            }
            lam_prev = lam;
            f_prev   = f_curr;
        }
    }

    if (roots.empty()) {
        throw std::runtime_error(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "no determinant zeros found in the search window");
    }

    // Each n >= 1 mode is doubly degenerate (cos and sin in theta).
    // Expand the root list with one extra copy per root with n >= 1.
    std::vector<double> omegas;
    omegas.reserve(roots.size() * 2);
    for (const auto& r : roots) {
        const double omega = r.lambda * r.lambda * omega_per_lambda2;
        omegas.push_back(omega);
        if (r.n >= 1) {
            omegas.push_back(omega);
        }
    }

    if (omegas.size() < n_modes) {
        throw std::runtime_error(
            "annular_plate_clamped_clamped_angular_frequencies: "
            "search window did not capture enough modes; consider "
            "smaller n_modes or report the geometry as a bug");
    }

    std::partial_sort(
        omegas.begin(),
        omegas.begin() + static_cast<std::ptrdiff_t>(n_modes),
        omegas.end());
    omegas.resize(n_modes);
    return omegas;
}

}  // namespace chladni::analytical
