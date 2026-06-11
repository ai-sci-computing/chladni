#pragma once

/**
 * @file shell.hpp
 * @brief Closed-form analytical references for thin-shell vibration
 *        (cylindrical and complete-spherical).
 *
 * The simply-supported (shear-diaphragm, SD-SD) finite-length thin
 * circular cylindrical shell is the closest closed-form analogue to a
 * drum shell. We support the **Donnell-Mushtari** approximation of
 * eighth-order classical shell theory: of the available theories
 * (Donnell-Mushtari, Love-Timoshenko, Goldenveizer-Novozhilov, Flugge,
 * Sanders, Vlasov...) it is the simplest that gives sensible
 * frequencies for typical drum-shell aspect ratios and thicknesses;
 * other theories agree with it to <1% for thinness h/R << 1 and small
 * @f$ k = h^2 / (12 R^2) @f$.
 *
 * For the complete (closed) spherical shell, we ship Wilkinson's 1965
 * cubic in @f$ \lambda^2 @f$ which embeds classical bending plus
 * transverse-shear and rotary-inertia corrections. This is the
 * apples-to-apples reference for the closed-mesh FEM path
 * (`compute_shell_modes_loop` on a closed icosphere) — see
 * @ref chladni::analytical::complete_spherical_shell_wilkinson_angular_frequencies.
 *
 * @section ana_shell_refs References
 * - @cite leissa_1973_shells, ch. 2 (closed cylindrical shells —
 *   shear diaphragms at both ends), eqs. (2.35) and (2.36).
 * - @cite wilkinson_1965_spherical_shells (closed spherical shells,
 *   classical+shear cubic) and @cite duffey_2005_spherical_shells
 *   eqs. (1)-(3) (legible restatement and experimental benchmark).
 */

#include <chladni/material.hpp>

#include <cstddef>
#include <vector>

namespace chladni::analytical {

/**
 * @brief Geometry of a thin circular cylindrical shell.
 *
 * The shell is a straight tube of axial length @f$L@f$, mid-surface
 * radius @f$R@f$, and uniform thickness @f$h@f$. Donnell-Mushtari
 * theory applies for @f$h \ll R@f$; in practice @f$h/R \le 1/20@f$ is
 * the rule of thumb.
 */
struct CylindricalShell {
    double radius;     ///< Mid-surface radius @f$R@f$ (m), @f$> 0@f$.
    double length;     ///< Axial length @f$L@f$ (m), @f$> 0@f$.
    double thickness;  ///< Wall thickness @f$h@f$ (m), @f$> 0@f$.
};

/**
 * @brief Natural angular frequencies of a thin circular cylindrical shell
 *        supported at both ends by shear diaphragms (SD-SD), under the
 *        Donnell-Mushtari approximation.
 *
 * For each axial half-wave number @f$m = 1, 2, \ldots@f$ and circumferential
 * wave number @f$n = 0, 1, 2, \ldots@f$, the Donnell-Mushtari characteristic
 * equation yields a real cubic in the squared nondimensional frequency
 * @f$\Omega^2@f$:
 * @f[
 *    \Omega^6 \;-\; K_2 \,\Omega^4 \;+\; K_1 \,\Omega^2 \;-\; K_0 \;=\; 0,
 * @f]
 * with
 * @f[
 *    \begin{aligned}
 *      K_2 &= 1 + \tfrac{1}{2}(3 - \nu)(n^2 + \lambda^2) + k (n^2 + \lambda^2)^2, \\
 *      K_1 &= \tfrac{1}{2}(1-\nu)\bigl[(3 + 2\nu)\lambda^2 + n^2
 *             + (n^2 + \lambda^2)^2
 *             + \tfrac{3-\nu}{1-\nu}\, k (n^2 + \lambda^2)^3\bigr], \\
 *      K_0 &= \tfrac{1}{2}(1-\nu)\bigl[(1-\nu^2)\lambda^4 + k (n^2 + \lambda^2)^4\bigr],
 *    \end{aligned}
 * @f]
 * @f$ \lambda = m \pi R / L @f$, and @f$ k = h^2 / (12 R^2) @f$ is the
 * thinness parameter. The three real roots correspond to predominantly
 * radial (flexural), axial, and circumferential (torsional) motion;
 * the smallest is usually radial. The dimensional angular frequency is
 * @f$ \omega = (\Omega / R) \sqrt{ E / (\rho(1-\nu^2)) } @f$.
 *
 * Reference: @cite leissa_1973_shells, eqs. (2.26), (2.35), (2.36).
 *
 * @param geom      Shell geometry (R, L, h all @f$> 0@f$).
 * @param material  Linear-elastic material parameters
 *                  (@f$E > 0@f$, @f$-1 < \nu < 1/2@f$, @f$\rho > 0@f$).
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *
 * @return Ascending list of length @p n_modes of natural angular
 *         frequencies @f$\omega@f$ in rad/s. Frequencies that the
 *         cubic returns as zero (rigid-body modes) or imaginary
 *         (Donnell-Mushtari is known to give a spurious imaginary
 *         frequency for @f$n = 1@f$ rigid-body lateral translation)
 *         are filtered out.
 *
 * @throws std::invalid_argument
 *         on non-positive geometry, non-physical material, or zero
 *         @p n_modes.
 */
std::vector<double> simply_supported_cylindrical_shell_donnell_mushtari_angular_frequencies(
    const CylindricalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes);

/**
 * @brief Natural angular frequencies of a thin circular cylindrical
 *        shell with **completely free** (unconstrained) ends, in the
 *        inextensional approximation of classical shell theory.
 *
 * Free-free is the boundary condition produced by the discrete-shells
 * FEM (see @ref chladni::shell::compute_shell_modes), so this is the
 * apples-to-apples analytical reference for that solver.
 *
 * Within Rayleigh's inextensional hypothesis (the mid-surface deforms
 * without stretching) the deformation is a superposition of cross-
 * sectional ovalling, breathing, and flexure modes. For each
 * circumferential wave number @f$ n \ge 2 @f$ Love (1888) gave the
 * finite-length frequency
 * @f[
 *   \omega_n^2 \;=\; \frac{D}{\rho h R^4}\,\frac{n^2 (n^2-1)^2}{n^2+1}
 *                     \cdot
 *                     \frac{\,1 \;+\; \dfrac{24(1-\nu) R^2}{n^2\,L^2}\,}
 *                          {\,1 \;+\; \dfrac{12 R^2}{n^2(n^2+1)\,L^2}\,}
 *                     \,,
 * @f]
 * which reduces to Rayleigh's formula
 * @f$\omega_n^2 = (D/\rho h R^4)\,n^2(n^2-1)^2/(n^2+1)@f$
 * as @f$L \to \infty@f$. Here @f$D = E h^3 / (12(1-\nu^2))@f$ is the
 * plate flexural rigidity and @f$\rho h@f$ is the mid-surface mass
 * density.
 *
 * @section inext_caveats Caveats
 *  - @f$n = 0@f$ (uniform breathing) is *extensional* — not handled here.
 *  - @f$n = 1@f$ is rigid-body translation — frequency 0, omitted.
 *  - The first @f$n@f$ included is therefore @f$n = 2@f$ (ovalling).
 *  - Inextensional is accurate to <5% for typical thin-shell aspect
 *    ratios (h/R << 1, L/R >~ 5). For shorter shells the membrane-
 *    extension contribution stiffens the modes and the FEM frequency
 *    will exceed this analytical value. For an apples-to-apples FEM
 *    comparison test, generate a long thin cylinder
 *    (e.g. L/R >= 20, h/R <= 1/100).
 *  - Each returned frequency is a doubly-degenerate pair in the FEM
 *    eigensolve (cos- and sin-phase ovalling at the same @f$n@f$).
 *
 * Reference: @cite leissa_1973_shells §2.4.5, eqs. (2.130) Rayleigh
 * and (2.132) Love.
 *
 * @param geom      Shell geometry (R, L, h all @f$> 0@f$).
 *                  L only enters via the Love correction.
 * @param material  Linear-elastic material parameters
 *                  (@f$E > 0@f$, @f$-1 < \nu < 1/2@f$, @f$\rho > 0@f$).
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *                  The k-th returned frequency (0-based) corresponds to
 *                  circumferential wave number @f$n = k + 2@f$: the n=0
 *                  breathing and n=1 rigid-body modes are omitted, so the
 *                  first physical inextensional mode is n=2 (consistent with
 *                  the @f$n = 2, 3, \ldots@f$ return order below).
 *
 * @return Length-@p n_modes vector of natural angular frequencies (rad/s),
 *         in ascending circumferential-wave-number order
 *         (@f$ n = 2, 3, \ldots, n_\text{modes}+1 @f$). Frequencies
 *         themselves are ascending in @f$ n @f$ for thin shells.
 *
 * @throws std::invalid_argument
 *         on non-positive geometry, non-physical material, or zero
 *         @p n_modes.
 */
std::vector<double> free_free_cylindrical_shell_inextensional_angular_frequencies(
    const CylindricalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes);

/**
 * @brief Geometry of a thin complete (closed) spherical shell.
 *
 * The shell is a hollow sphere of mid-surface radius @f$R@f$ and
 * uniform wall thickness @f$h@f$. Wilkinson's classical+shear theory
 * applies for @f$h \ll R@f$; in practice the @f$R/h \ge 20@f$ rule of
 * thumb covers all musically interesting drum-shell shells.
 */
struct SphericalShell {
    double radius;     ///< Mid-surface radius @f$R@f$ (m), @f$> 0@f$.
    double thickness;  ///< Wall thickness @f$h@f$ (m), @f$> 0@f$.
};

/**
 * @brief Natural angular frequencies of a thin complete (closed)
 *        spherical shell, axisymmetric flexural-branch modes, under
 *        Wilkinson's 1965 classical-plus-shear theory.
 *
 * For a complete spherical shell the modal spectrum splits into two
 * infinite branches at every mode index @f$ n \ge 2 @f$ — the lower
 * (predominantly bending) branch and an upper (predominantly
 * membrane / extensional) branch — plus a third much higher branch
 * dominated by transverse shear. The three branches are the three
 * roots of a cubic in the squared nondimensional frequency
 * @f$ \lambda^2 @f$,
 * @f[
 *   \alpha\,\lambda^6 \;-\; \beta\,\lambda^4 \;+\; \delta\,\lambda^2 \;-\; \gamma \;=\; 0,
 * @f]
 * where (@cite duffey_2005_spherical_shells eqs. (1)-(3))
 * @f[
 *   \begin{aligned}
 *     \alpha &= \tfrac{2 k_s k_1 (k_r k_1 - c_r c_1)}{1 - \nu}, \\
 *     \beta  &= (k_r k_1 - c_r c_1)\!\left[r + \tfrac{4 k_s (1+\nu)}{1-\nu}\right] \\
 *            &\quad + k_1\!\left[\xi(k_1 + c_1) + c_r + k_r
 *                               + 2 k_s (k_1 + k_r)\!\left(\tfrac{r}{1-\nu} - 1\right)\right], \\
 *     \delta &= (\xi c_1 + c_r)(1+\nu)(2 - r) \\
 *            &\quad + k_r\bigl[r(r - 3 - \nu) + 2(1+\nu)\bigr]\!\bigl[(r-2) k_s + 1\bigr] \\
 *            &\quad + k_1\!\left[\tfrac{2 k_r r (r + 4\nu)}{1-\nu}
 *                               + r(r + \xi + \nu) + (1 + 3\nu)(\xi - 2 k_s) - (1-\nu)\right], \\
 *     \gamma &= (r - 2)\!\left[r(r - 2) + 2 k_s (1+\nu)(r - 1 + \nu) + (1 - \nu^2)(\xi + 1)\right],
 *   \end{aligned}
 * @f]
 * with mode-index polynomial @f$ r = n(n+1) @f$, thinness parameter
 * @f$ k = h^2 / (12 R^2) @f$, @f$ \xi = 1/k @f$,
 * @f$ k_1 = 1 + k @f$, @f$ k_r = 1 + 1.8 k @f$, @f$ c_1 = 2 k @f$,
 * @f$ c_r = 2 @f$, and Reissner-Mindlin shear correction
 * @f$ k_s \approx 1.2 @f$. The dimensional angular frequency for a
 * given root @f$ \lambda^2 @f$ is
 * @f[
 *    \omega \;=\; \frac{\lambda}{R}\,\sqrt{\frac{E}{\rho (1-\nu^2)}}\,.
 * @f]
 *
 * This routine returns the @b lower-branch (bending-dominated) root
 * @f$ \lambda^2 @f$ for every @f$ n = 2, 3, \ldots @f$ in ascending
 * order. The lower branch is what an FEM thin-shell solver
 * (`chladni::shell::compute_shell_modes_loop`) reproduces in its
 * lowest non-rigid modes; the upper (membrane) branch sits a factor
 * of @f$ \mathcal{O}(R/h) @f$ higher and is irrelevant for typical
 * drum-shell-thickness audio applications. Each returned frequency is
 * @f$ (2n+1) @f$-fold degenerate on a perfect sphere (one
 * axisymmetric mode + @f$ 2n @f$ non-axisymmetric of the same
 * frequency); the FEM eigensolve will return that many near-equal
 * eigenvalues per @f$ n @f$, so caller-side mode identification is by
 * frequency cluster.
 *
 * @section sd_caveats Caveats
 *  - @f$ n = 0 @f$ (uniform breathing) is purely extensional — sits
 *    on the upper branch and is omitted here. @f$ n = 1 @f$ is
 *    rigid-body translation, also omitted.
 *  - The first @f$ n @f$ included is therefore @f$ n = 2 @f$ (the
 *    spheroidal "American football" mode).
 *  - Wilkinson's theory is accurate to <1% on @f$ R/h \ge 50 @f$
 *    (Duffey's spherical-float benchmark, @f$ R/h = 71.5 @f$, agrees
 *    to within 0.02-0.73% over the lowest four detected modes).
 *  - For comparison against an FEM closed-sphere computation, expect
 *    larger relative errors (single-digit-percent at moderate mesh
 *    refinement) because the FEM has its own discretisation error on
 *    top of the analytic theory's truncation error. Use the
 *    multi-pass icosphere fixture
 *    (`chladni::mesh::generate_icosphere(R, k=2)` + `n_passes=2`)
 *    for a tight comparison.
 *
 * Reference: @cite wilkinson_1965_spherical_shells (original cubic);
 * @cite duffey_2005_spherical_shells eqs. (1)-(3) (legible restatement
 * and the experimental benchmark used in unit tests).
 *
 * @param geom      Shell geometry (R, h both @f$> 0@f$).
 * @param material  Linear-elastic material parameters
 *                  (@f$E > 0@f$, @f$-1 < \nu < 1/2@f$, @f$\rho > 0@f$).
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *                  The k-th returned frequency corresponds to spherical
 *                  mode index @f$n = k + 2@f$.
 *
 * @return Length-@p n_modes vector of natural angular frequencies (rad/s),
 *         ascending in @f$ n @f$ (which for thin shells is also
 *         ascending in @f$ \omega @f$ on the lower branch).
 *
 * @throws std::invalid_argument
 *         on non-positive geometry, non-physical material, or zero
 *         @p n_modes.
 * @throws std::runtime_error
 *         if for some @f$ n @f$ the cubic fails to yield a positive
 *         real lower-branch root (only seen for non-physical inputs).
 */
std::vector<double>
complete_spherical_shell_wilkinson_angular_frequencies(
    const SphericalShell& geom,
    const ::chladni::IsotropicMaterial& material,
    std::size_t n_modes);

}  // namespace chladni::analytical
