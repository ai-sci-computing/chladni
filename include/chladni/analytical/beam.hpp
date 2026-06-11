#pragma once

/**
 * @file beam.hpp
 * @brief Closed-form analytical references for thin beam vibration.
 *
 * Euler-Bernoulli beam theory describes the transverse vibration of a
 * slender prismatic bar by the fourth-order PDE
 * @f[
 *   E\,I\,\frac{\partial^{4} w}{\partial x^{4}}
 *   + \rho\,A\,\frac{\partial^{2} w}{\partial t^{2}} \;=\; 0,
 * @f]
 * which separates into harmonic modes
 * @f$ w_n(x, t) = \phi_n(x)\,\cos(\omega_n t) @f$ with the spatial
 * shape satisfying @f$ \phi_n^{(4)}(x) = \beta_n^{4}\,\phi_n(x) @f$.
 * The eigenvalue @f$ \beta_n L @f$ depends only on the boundary
 * conditions, while the dimensional frequency follows from
 * @f$ \omega_n^{2} = (\beta_n L)^{4} \, E\,I\,/\,(\rho\,A\,L^{4}) @f$.
 *
 * For a rectangular cross-section of width @f$w@f$ and thickness
 * @f$h@f$ the cross-section quantities are @f$A = wh@f$ and
 * @f$I = w h^{3} / 12@f$ (second moment about the neutral axis
 * perpendicular to the bending direction).
 *
 * Beam theory is asymptotically valid for @f$L \gg \max(w, h)@f$;
 * an aspect ratio of @f$L / \max(w, h) \gtrsim 10@f$ is the usual
 * rule of thumb. Below that, Timoshenko (shear) corrections start
 * to matter.
 *
 * @section beam_refs References
 * - Standard derivation in any vibrations textbook;
 *   Inman *Engineering Vibration* (4th ed.) ch. 6 and Rao
 *   *Mechanical Vibrations* ch. 8 are typical sources for the
 *   @f$\beta_n L@f$ tabulation.
 */

#include <chladni/material.hpp>

#include <cstddef>
#include <vector>

namespace chladni::analytical {

/**
 * @brief Geometry of a uniform prismatic beam with rectangular
 *        cross-section.
 */
struct RectangularBeam {
    double length;     ///< Beam length @f$L@f$ (m), @f$> 0@f$.
    double width;      ///< Cross-section width @f$w@f$ (m), @f$> 0@f$.
    double thickness;  ///< Cross-section thickness @f$h@f$ (m), @f$> 0@f$;
                       ///< this is the dimension along the bending direction.
};

/**
 * @brief Dimensionless eigenvalues @f$\beta_n L@f$ for a
 *        free-free Euler-Bernoulli beam, in ascending order.
 *
 * Free-free is the boundary condition of an unconstrained bar (zero
 * shear and zero bending moment at both ends). The eigenvalue
 * equation reduces to the transcendental
 * @f[
 *   \cos(\beta L)\,\cosh(\beta L) \;=\; 1,
 * @f]
 * with roots @f$\beta_n L \approx 4.7300, 7.8532, 10.9956, 14.1372,
 * 17.2788, \ldots@f$. The trivial root at @f$0@f$ corresponds to a
 * rigid-body translation/rotation and is excluded.
 *
 * @param n_modes  Number of roots to return; must be @f$\ge 1@f$.
 * @return         Length-@p n_modes vector of universal roots
 *                 @f$\beta_n L@f$, ascending, accurate to roughly
 *                 @f$10^{-12}@f$ (limited by bisection convergence).
 *
 * @throws std::invalid_argument if @p n_modes is zero.
 */
std::vector<double> free_free_beam_eigenvalue_roots(std::size_t n_modes);

/**
 * @brief Natural angular frequencies of a free-free Euler-Bernoulli
 *        beam with rectangular cross-section.
 *
 * For each universal root @f$\beta_n L@f$ from
 * @ref free_free_beam_eigenvalue_roots, the dimensional angular
 * frequency follows from
 * @f[
 *   \omega_n^{2} \;=\; (\beta_n L)^{4}\,
 *     \frac{E\,I}{\rho\,A\,L^{4}},
 * @f]
 * with @f$A = w h@f$ and @f$I = w h^{3}/12@f$.
 *
 * @param geom      Beam geometry (all dimensions @f$> 0@f$).
 * @param material  Linear-elastic material parameters
 *                  (@f$E > 0@f$, @f$\rho > 0@f$; @f$\nu@f$ is unused
 *                  by Euler-Bernoulli theory but the field must
 *                  still satisfy the usual physical constraint).
 * @param n_modes   Number of frequencies to return; must be @f$\ge 1@f$.
 *
 * @return Ascending list of length @p n_modes of natural angular
 *         frequencies @f$\omega_n@f$ (rad/s).
 *
 * @throws std::invalid_argument
 *         on non-positive geometry, non-physical material,
 *         or zero @p n_modes.
 */
std::vector<double> free_free_beam_angular_frequencies(
    const RectangularBeam& geom,
    const chladni::IsotropicMaterial& material,
    std::size_t n_modes);

}  // namespace chladni::analytical
