/**
 * @file shell.cpp
 * @brief Implementations of the analytical thin-shell references.
 *
 * For the simply-supported cylindrical shell, solves the Donnell-Mushtari
 * characteristic cubic in @f$ \Omega^2 @f$ for every @f$(m, n)@f$ pair
 * in a finite enumeration window, sorts the resulting positive real
 * frequencies, and returns the smallest @p n_modes. The cubic is
 * solved by the eigenvalues of its 3x3 companion matrix; for the
 * thin-shell regime all three roots are real (and the smallest is the
 * predominantly radial flexural mode).
 * Reference: @cite leissa_1973_shells, eqs. (2.26), (2.35), (2.36).
 *
 * For the complete (closed) spherical shell, solves Wilkinson's cubic
 * in @f$ \lambda^2 @f$ for each @f$ n = 2, 3, \ldots @f$ and returns
 * the smallest positive real root (lower / bending branch) per @f$ n @f$.
 * Reference: @cite wilkinson_1965_spherical_shells; legible restatement
 * in @cite duffey_2005_spherical_shells eqs. (1)-(3).
 */

#include <chladni/analytical/shell.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace chladni::analytical {

namespace {

constexpr double pi = std::numbers::pi_v<double>;

void validate_inputs(const CylindricalShell& geom,
                     const ::chladni::IsotropicMaterial& material,
                     std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "simply_supported_cylindrical_shell_donnell_mushtari_"
            "angular_frequencies: n_modes must be >= 1");
    }
    if (geom.radius <= 0.0 || geom.length <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "simply_supported_cylindrical_shell_donnell_mushtari_"
            "angular_frequencies: all geometric dimensions must be > 0");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "simply_supported_cylindrical_shell_donnell_mushtari_"
            "angular_frequencies: Young's modulus and density must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "simply_supported_cylindrical_shell_donnell_mushtari_"
            "angular_frequencies: Poisson's ratio must lie in (-1, 1/2)");
    }
}

/**
 * @brief Solve x^3 - K2 x^2 + K1 x - K0 = 0 via the 3x3 companion matrix.
 *
 * The 3x3 companion matrix of x^3 + a2 x^2 + a1 x + a0 = 0 is
 *
 *     [ 0  0  -a0 ]
 *     [ 1  0  -a1 ]
 *     [ 0  1  -a2 ]
 *
 * with a2 = -K2, a1 = K1, a0 = -K0 in our case.
 */
std::array<std::complex<double>, 3>
solve_cubic_in_omega_squared(double K2, double K1, double K0)
{
    Eigen::Matrix3d C;
    C << 0.0, 0.0,  K0,
         1.0, 0.0, -K1,
         0.0, 1.0,  K2;
    Eigen::EigenSolver<Eigen::Matrix3d> solver(C, /*computeEigenvectors=*/false);
    const auto ev = solver.eigenvalues();
    return {ev(0), ev(1), ev(2)};
}

}  // anonymous namespace

std::vector<double>
simply_supported_cylindrical_shell_donnell_mushtari_angular_frequencies(
    const CylindricalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    validate_inputs(geom, material, n_modes);

    const double R   = geom.radius;
    const double L   = geom.length;
    const double h   = geom.thickness;
    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;

    const double k_thinness = h * h / (12.0 * R * R);
    // omega = (Omega / R) * sqrt(E / (rho (1 - nu^2)))   (Leissa eq. 2.26).
    const double c_omega = (1.0 / R) * std::sqrt(E / (rho * (1.0 - nu * nu)));

    // Heuristic enumeration window. Lowest cylindrical-shell modes for
    // any reasonable aspect ratio sit in low m, low-to-mid n. We allow
    // generously to keep the routine robust to unusual h/R or L/R.
    const auto sqrt_n = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(n_modes))));
    const std::size_t M_max = std::max<std::size_t>(8u, 2u + 2u * sqrt_n);
    const std::size_t N_max = std::max<std::size_t>(20u, 4u * sqrt_n);

    std::vector<double> omegas;
    omegas.reserve(M_max * (N_max + 1) * 3);

    for (std::size_t m = 1; m <= M_max; ++m) {
        const double lam = static_cast<double>(m) * pi * R / L;
        const double l2 = lam * lam;
        const double l4 = l2 * l2;
        for (std::size_t n_idx = 0; n_idx <= N_max; ++n_idx) {
            const auto n_d = static_cast<double>(n_idx);
            const double n2 = n_d * n_d;
            const double s  = n2 + l2;
            const double s2 = s * s;
            const double s3 = s2 * s;
            const double s4 = s2 * s2;

            // Leissa eq. (2.36), Donnell-Mushtari row. The thinness term
            // in K1 carries a (3-nu)/(1-nu) factor inside the bracket:
            // K1 is the sum of the principal 2x2 minors of the DM modal
            // matrix, whose only k-dependent contribution is
            // (A11 + A22) k s^2 = (3-nu)/2 k s^3.
            const double K2 = 1.0 + 0.5 * (3.0 - nu) * s + k_thinness * s2;
            const double K1 = 0.5 * (1.0 - nu)
                * ((3.0 + 2.0 * nu) * l2 + n2 + s2
                   + (3.0 - nu) / (1.0 - nu) * k_thinness * s3);
            const double K0 = 0.5 * (1.0 - nu)
                * ((1.0 - nu * nu) * l4 + k_thinness * s4);

            const auto roots = solve_cubic_in_omega_squared(K2, K1, K0);
            for (const auto& r : roots) {
                // Discard imaginary roots (Donnell-Mushtari is known
                // to give a spurious imaginary value for some n=1
                // configurations) and the rigid-body zero modes.
                const double im_tol = 1.0e-8 * (std::abs(r.real()) + 1.0);
                if (std::abs(r.imag()) > im_tol) {
                    continue;
                }
                if (r.real() < 1.0e-12) {
                    continue;
                }
                omegas.push_back(std::sqrt(r.real()) * c_omega);
            }
        }
    }

    if (omegas.size() < n_modes) {
        throw std::runtime_error(
            "simply_supported_cylindrical_shell_donnell_mushtari_"
            "angular_frequencies: enumeration window did not capture "
            "enough physical modes; the geometry may be far from the "
            "thin-shell regime, or n_modes is unreasonably large");
    }

    std::partial_sort(
        omegas.begin(),
        omegas.begin() + static_cast<std::ptrdiff_t>(n_modes),
        omegas.end());
    omegas.resize(n_modes);
    return omegas;
}

std::vector<double> free_free_cylindrical_shell_inextensional_angular_frequencies(
    const CylindricalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "free_free_cylindrical_shell_inextensional_angular_frequencies: "
            "n_modes must be >= 1");
    }
    if (geom.radius <= 0.0 || geom.length <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "free_free_cylindrical_shell_inextensional_angular_frequencies: "
            "shell radius, length and thickness must all be positive");
    }
    if (material.youngs_modulus <= 0.0) {
        throw std::invalid_argument(
            "free_free_cylindrical_shell_inextensional_angular_frequencies: "
            "Young's modulus must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "free_free_cylindrical_shell_inextensional_angular_frequencies: "
            "Poisson's ratio must lie in (-1, 1/2)");
    }
    if (material.density <= 0.0) {
        throw std::invalid_argument(
            "free_free_cylindrical_shell_inextensional_angular_frequencies: "
            "density must be > 0");
    }

    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;
    const double R   = geom.radius;
    const double L   = geom.length;
    const double h   = geom.thickness;

    // D = E h^3 / (12(1-nu^2)). Rayleigh's prefactor D/(rho h R^4).
    const double one_minus_nu2 = 1.0 - nu * nu;
    const double D = E * h * h * h / (12.0 * one_minus_nu2);
    const double prefactor = D / (rho * h * R * R * R * R);

    std::vector<double> omegas;
    omegas.reserve(n_modes);
    // k-th returned mode corresponds to circumferential wave number
    // n = k + 2 (n=0 is uniform breathing, n=1 is rigid-body translation,
    // both omitted; the first physical inextensional mode is n=2).
    for (std::size_t k = 0; k < n_modes; ++k) {
        const double n = static_cast<double>(k + 2);
        const double n2 = n * n;
        const double rayleigh_factor = n2 * (n2 - 1.0) * (n2 - 1.0)
                                     / (n2 + 1.0);

        // Love's finite-length correction (eq. 2.132). Reduces to 1 as
        // L -> infinity, recovering Rayleigh's eq. (2.130). Note the
        // SP-288 print of eq. (2.132) drops the square on (n^2-1) in
        // the LEADING factor — a typo; the square is required for the
        // stated l/R -> infinity reduction to eq. (2.130), so
        // rayleigh_factor above keeps it.
        const double R2_over_L2 = (R * R) / (L * L);
        const double love_num = 1.0
            + 24.0 * (1.0 - nu) * R2_over_L2 / n2;
        const double love_den = 1.0
            + 12.0 * R2_over_L2 / (n2 * (n2 + 1.0));
        const double omega2 = prefactor * rayleigh_factor * (love_num / love_den);

        omegas.push_back(std::sqrt(omega2));
    }
    return omegas;
}

std::vector<double>
complete_spherical_shell_wilkinson_angular_frequencies(
    const SphericalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "complete_spherical_shell_wilkinson_angular_frequencies: "
            "n_modes must be >= 1");
    }
    if (geom.radius <= 0.0 || geom.thickness <= 0.0) {
        throw std::invalid_argument(
            "complete_spherical_shell_wilkinson_angular_frequencies: "
            "radius and thickness must both be > 0");
    }
    if (material.youngs_modulus <= 0.0 || material.density <= 0.0) {
        throw std::invalid_argument(
            "complete_spherical_shell_wilkinson_angular_frequencies: "
            "Young's modulus and density must be > 0");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "complete_spherical_shell_wilkinson_angular_frequencies: "
            "Poisson's ratio must lie in (-1, 1/2)");
    }

    const double R   = geom.radius;
    const double h   = geom.thickness;
    const double E   = material.youngs_modulus;
    const double nu  = material.poisson_ratio;
    const double rho = material.density;

    // Auxiliary constants — Duffey eqs. (3).
    const double k        = h * h / (12.0 * R * R);
    const double xi       = 1.0 / k;
    const double k1       = 1.0 + k;
    const double kr       = 1.0 + 1.8 * k;
    const double c1       = 2.0 * k;
    const double cr       = 2.0;
    constexpr double ks   = 1.2;          // Reissner-Mindlin shear correction.
    const double one_m_nu = 1.0 - nu;
    const double one_p_nu = 1.0 + nu;
    const double one_m_nu2 = 1.0 - nu * nu;
    const double krk1_minus_crc1 = kr * k1 - cr * c1;

    // Dimensional prefactor: omega = (lambda / R) * sqrt(E / (rho (1-nu^2))).
    const double c_omega = (1.0 / R) * std::sqrt(E / (rho * one_m_nu2));

    std::vector<double> omegas;
    omegas.reserve(n_modes);

    for (std::size_t k_idx = 0; k_idx < n_modes; ++k_idx) {
        const double n_d = static_cast<double>(k_idx + 2);  // skip n=0,1
        const double r   = n_d * (n_d + 1.0);

        // Duffey eq. (2).
        const double alpha = 2.0 * ks * k1 * krk1_minus_crc1 / one_m_nu;

        const double beta = krk1_minus_crc1 *
            (r + 4.0 * ks * one_p_nu / one_m_nu)
            + k1 * (xi * (k1 + c1) + cr + kr
                    + 2.0 * ks * (k1 + kr) * (r / one_m_nu - 1.0));

        const double delta = (xi * c1 + cr) * one_p_nu * (2.0 - r)
            + kr * (r * (r - 3.0 - nu) + 2.0 * one_p_nu)
                 * ((r - 2.0) * ks + 1.0)
            + k1 * (2.0 * kr * r * (r + 4.0 * nu) / one_m_nu
                    + r * (r + xi + nu)
                    + (1.0 + 3.0 * nu) * (xi - 2.0 * ks)
                    - one_m_nu);

        const double gamma = (r - 2.0)
            * (r * (r - 2.0)
               + 2.0 * ks * one_p_nu * (r - 1.0 + nu)
               + one_m_nu2 * (xi + 1.0));

        // Cubic in x = lambda^2:  alpha x^3 - beta x^2 + delta x - gamma = 0
        // -> monic: x^3 + (-beta/alpha) x^2 + (delta/alpha) x + (-gamma/alpha) = 0
        // Companion matrix of x^3 + a2 x^2 + a1 x + a0 with eigenvalues equal
        // to the roots is built as in solve_cubic_in_omega_squared above.
        const auto roots = solve_cubic_in_omega_squared(
            /*K2=*/beta / alpha,
            /*K1=*/delta / alpha,
            /*K0=*/gamma / alpha);

        // Smallest positive real root = lower (bending) branch.
        double x_min = std::numeric_limits<double>::infinity();
        for (const auto& root : roots) {
            const double im_tol = 1.0e-8 * (std::abs(root.real()) + 1.0);
            if (std::abs(root.imag()) > im_tol) {
                continue;
            }
            if (root.real() <= 0.0) {
                continue;
            }
            x_min = std::min(x_min, root.real());
        }
        if (!std::isfinite(x_min)) {
            throw std::runtime_error(
                "complete_spherical_shell_wilkinson_angular_frequencies: "
                "no positive real root for n=" + std::to_string(static_cast<int>(n_d))
                + "; the cubic does not yield a physical lower-branch "
                  "frequency for these inputs");
        }
        omegas.push_back(std::sqrt(x_min) * c_omega);
    }
    return omegas;
}

}  // namespace chladni::analytical
