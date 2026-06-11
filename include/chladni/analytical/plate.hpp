#pragma once

/**
 * @file plate.hpp
 * @brief Closed-form analytical references for thin-plate vibration.
 *
 * This header provides ground-truth natural frequencies for canonical
 * thin-plate configurations under Kirchhoff plate theory. They are used
 * by the test suite to pin the FEM solver against known results, and by
 * demos that compare simulated vs. analytical spectra.
 *
 * All frequencies returned by these functions are angular frequencies
 * @f$\omega@f$ in radians per second; convert to Hertz with
 * @f$f = \omega / (2\pi)@f$.
 *
 * @section plate_refs References
 * - @cite leissa_1969_plates — primary source for plate vibration formulas.
 */

#include <chladni/material.hpp>

#include <cstddef>
#include <vector>

namespace chladni::analytical {

/**
 * @brief Geometry of a rectangular plate.
 *
 * The plate occupies the rectangle @f$[0,a] \times [0,b]@f$ in the
 * mid-plane and has uniform thickness @f$h@f$ along the normal.
 * Thin-plate (Kirchhoff) theory assumes @f$h \ll \min(a,b)@f$;
 * a ratio of @f$h / \min(a,b) \le 1/20@f$ is a usual rule of thumb.
 */
struct RectanglePlate {
    double length_a;   ///< Extent along the x-axis @f$a@f$ (m), @f$> 0@f$.
    double length_b;   ///< Extent along the y-axis @f$b@f$ (m), @f$> 0@f$.
    double thickness;  ///< Plate thickness @f$h@f$ (m), @f$> 0@f$.
};

/**
 * @brief Geometry of a uniform solid (no hole) circular plate.
 */
struct CircularPlate {
    double radius;     ///< Plate radius @f$a@f$ (m), @f$> 0@f$.
    double thickness;  ///< Plate thickness @f$h@f$ (m), @f$> 0@f$.
};

/**
 * @brief Natural angular frequencies of a thin circular plate with a
 *        completely free edge (Kirchhoff thin-plate theory).
 *
 * This is the configuration of Chladni's classical experiments — a
 * disk free to vibrate, sprinkled with sand to visualise nodal
 * patterns. For each circumferential wave number @f$n \ge 0@f$ the
 * radial deflection is
 * @f$ W_n(r) = A\,J_n(kr) + C\,I_n(kr) @f$ (the @f$Y_n@f$ and
 * @f$K_n@f$ branches are dropped for finiteness at the centre of a
 * solid plate), with @f$ k^{4} = \rho h\,\omega^{2}/D @f$. Imposing
 * the free-edge conditions @f$M_r(a) = 0@f$ and the Kelvin-Kirchhoff
 * edge reaction @f$V_r(a) = 0@f$ collapses to the 2x2 frequency
 * determinant
 * @f[
 *   M_J(\lambda)\,V_I(\lambda) \;-\; M_I(\lambda)\,V_J(\lambda) = 0,
 * @f]
 * with @f$\lambda = ka@f$ and
 * @f{align*}{
 *   M_J(\lambda) &= \lambda^{2}J_n(\lambda)
 *                 + (1{-}\nu)\bigl[\lambda\,J_n'(\lambda)
 *                                  - n^{2}J_n(\lambda)\bigr], \\
 *   M_I(\lambda) &= \lambda^{2}I_n(\lambda)
 *                 - (1{-}\nu)\bigl[\lambda\,I_n'(\lambda)
 *                                  - n^{2}I_n(\lambda)\bigr], \\
 *   V_J(\lambda) &= \lambda^{3}J_n'(\lambda)
 *                 + (1{-}\nu)\,n^{2}\bigl[\lambda\,J_n'(\lambda)
 *                                          - J_n(\lambda)\bigr], \\
 *   V_I(\lambda) &= \lambda^{3}I_n'(\lambda)
 *                 - (1{-}\nu)\,n^{2}\bigl[\lambda\,I_n'(\lambda)
 *                                          - I_n(\lambda)\bigr].
 * @f}
 * The dimensional angular frequency follows from the eigenvalue
 * @f$\lambda@f$ as
 * @f$ \omega = (\lambda^{2}/a^{2})\sqrt{D/(\rho h)} @f$, with
 * @f$D = E h^{3} / (12 (1{-}\nu^{2}))@f$.
 *
 * @section degeneracy Mode degeneracy and exclusions
 * - @f$n = 0@f$ (axisymmetric): non-degenerate; only @f$\cos(0\theta) = 1@f$.
 *   The trivial @f$\lambda = 0@f$ root corresponds to rigid-body
 *   translation and is excluded.
 * - @f$n = 1@f$: doubly degenerate (cos and sin in @f$\theta@f$).
 *   @f$\lambda = 0@f$ root corresponds to rigid-body rotation about a
 *   diameter — excluded.
 * - @f$n \ge 2@f$: doubly degenerate.
 *
 * Reference: @cite leissa_1969_plates §2.1.3 "Completely Free
 * Plates" — eq. (2.14) gives the same characteristic equation
 * (rearranged) and Table 2.5 tabulates @f$\lambda^{2}@f$ at
 * @f$\nu = 0.33@f$.
 *
 * @param geom      Plate geometry.
 * @param material  Linear-elastic material parameters.
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *
 * @return Ascending list of length @p n_modes containing the lowest
 *         natural angular frequencies @f$\omega_{ns}@f$, in rad/s,
 *         with each @f$n \ge 1@f$ root expanded to its degenerate
 *         pair.
 *
 * @throws std::invalid_argument
 *         on non-positive geometry, non-physical material, or zero
 *         @p n_modes.
 * @throws std::runtime_error
 *         if the determinant root finder fails to capture enough
 *         modes within its search window.
 */
std::vector<double> free_edge_circular_plate_angular_frequencies(
    const CircularPlate& geom,
    const chladni::IsotropicMaterial& material,
    std::size_t n_modes);

/**
 * @brief Geometry of a uniform annular (ring) plate.
 *
 * The plate is a flat ring with outer radius @f$a@f$, inner radius
 * @f$b@f$, and uniform thickness @f$h@f$ along the normal.
 * Kirchhoff thin-plate theory assumes @f$h \ll a - b@f$.
 */
struct AnnularPlate {
    double radius_outer;  ///< Outer radius @f$a@f$ (m), @f$> 0@f$.
    double radius_inner;  ///< Inner radius @f$b@f$ (m), @f$0 < b < a@f$.
    double thickness;     ///< Plate thickness @f$h@f$ (m), @f$> 0@f$.
};

/**
 * @brief Natural angular frequencies of a thin annular plate clamped on
 *        both the outer and inner edge (Kirchhoff thin-plate theory).
 *
 * For each circumferential wave number @f$n = 0, 1, 2, \ldots@f$ the
 * radial deflection is
 * @f[
 *   W_n(r) = A\,J_n(kr) + B\,Y_n(kr) + C\,I_n(kr) + D\,K_n(kr),
 * @f]
 * with @f$k^{4} = \rho h\,\omega^{2}/D@f$ and the flexural rigidity
 * @f$ D = E h^{3} / (12 (1 - \nu^{2})) @f$.
 * The clamped-clamped boundary conditions @f$ W_n(a) = W_n'(a) =
 * W_n(b) = W_n'(b) = 0 @f$ collapse to the 4x4 frequency determinant
 * @f[
 *   \det\!\begin{pmatrix}
 *     J_n(\lambda)        & Y_n(\lambda)        & I_n(\lambda)        & K_n(\lambda)        \\
 *     J_n(\alpha\lambda)  & Y_n(\alpha\lambda)  & I_n(\alpha\lambda)  & K_n(\alpha\lambda)  \\
 *     J_n'(\lambda)       & Y_n'(\lambda)       & I_n'(\lambda)       & K_n'(\lambda)       \\
 *     J_n'(\alpha\lambda) & Y_n'(\alpha\lambda) & I_n'(\alpha\lambda) & K_n'(\alpha\lambda)
 *   \end{pmatrix} = 0,
 * @f]
 * with @f$\lambda = ka@f$ and @f$\alpha = b/a@f$. The roots of this
 * determinant for each @f$n@f$, ordered by @f$\lambda@f$, give the
 * (n, s)-mode dimensionless frequency parameters
 * @f$\lambda_{ns}^{2} = \omega_{ns}\,a^{2}\sqrt{\rho h / D}@f$ from
 * which the angular frequencies follow as
 * @f$ \omega = (\lambda^{2}/a^{2}) \sqrt{D/(\rho h)} @f$.
 *
 * Modes with @f$ n \ge 1 @f$ are doubly degenerate (cos and sin in
 * @f$\theta@f$); both copies are returned in the spectrum.
 *
 * Reference: tabulated values from @cite leissa_1969_plates Table 2.18,
 * which reproduces @cite vogel_skinner_1965_annular with attribution.
 *
 * @param geom      Annular-plate geometry.
 * @param material  Linear-elastic material parameters.
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *
 * @return Ascending list of length @p n_modes containing the lowest
 *         natural angular frequencies @f$\omega_{ns}@f$, in rad/s.
 *
 * @throws std::invalid_argument
 *         on non-positive thickness, non-physical radii (e.g.
 *         @f$b \ge a@f$ or either @f$\le 0@f$), non-physical material,
 *         or zero @p n_modes.
 * @throws std::runtime_error
 *         if the determinant root finder fails to capture enough modes
 *         within its search window — in practice this should not happen
 *         for sensible geometries.
 */
std::vector<double> annular_plate_clamped_clamped_angular_frequencies(
    const AnnularPlate& geom,
    const chladni::IsotropicMaterial& material,
    std::size_t n_modes);

/**
 * @brief Natural angular frequencies of a thin rectangular plate that is
 *        simply-supported on all four edges (Kirchhoff thin-plate theory).
 *
 * Closed-form solution:
 * @f[
 *   \omega_{mn} \;=\; \pi^{2}\,\sqrt{\dfrac{D}{\rho\,h}}\,
 *                     \left( \dfrac{m^{2}}{a^{2}}
 *                          + \dfrac{n^{2}}{b^{2}} \right),
 *   \qquad m,n = 1,2,3,\ldots
 * @f]
 * with flexural rigidity @f$ D = \dfrac{E\, h^{3}}{12 \,(1 - \nu^{2})} @f$.
 * The associated mode shapes are
 * @f$ w_{mn}(x,y) = \sin(m\pi x / a) \, \sin(n\pi y / b) @f$.
 *
 * Reference: @cite leissa_1969_plates, ch. 4 (rectangular plates).
 *
 * @param geom      Plate geometry. All dimensions must be strictly positive.
 * @param material  Linear-elastic material parameters; must satisfy
 *                  @f$-1 < \nu < 1/2@f$ for physical admissibility and
 *                  @f$E > 0@f$, @f$\rho > 0@f$.
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *                  The routine enumerates enough @f$(m,n)@f$ candidates
 *                  internally to guarantee that the smallest @p n_modes
 *                  frequencies of the spectrum are captured.
 *
 * @return Ascending list of length @p n_modes containing the lowest
 *         natural angular frequencies @f$\omega_{mn}@f$, in rad/s.
 *
 * @throws std::invalid_argument
 *         if any geometric dimension is non-positive, if the material is
 *         non-physical, or if @p n_modes is zero.
 */
std::vector<double> simply_supported_rect_plate_angular_frequencies(
    const RectanglePlate& geom,
    const chladni::IsotropicMaterial& material,
    std::size_t n_modes);

}  // namespace chladni::analytical
