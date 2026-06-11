#pragma once

/**
 * @file material.hpp
 * @brief Shared material types for the chladni pipeline.
 *
 * The same isotropic, linearly-elastic material parameters drive every
 * stage — analytical references, FEM assembly (plates, shells, solids),
 * the resonator-bank synthesiser via Rayleigh damping, and the demo
 * applications. Keeping a single project-level type avoids ad-hoc
 * conversions and makes Doxygen cross-referencing trivial.
 */

namespace chladni {

/**
 * @brief Isotropic linearly elastic material parameters.
 *
 * Three numbers fully describe a small-strain isotropic linear-elastic
 * solid: Young's modulus @f$E@f$, Poisson's ratio @f$\nu@f$, and mass
 * density @f$\rho@f$. The shear modulus and Lamé parameters are
 * derivable: @f$ G = E / (2(1+\nu)) @f$,
 * @f$ \lambda = E\nu / ((1+\nu)(1-2\nu)) @f$, etc.
 *
 * @note Temperature dependence and viscoelasticity are out of scope
 *       for this struct. Damping is captured separately, typically as
 *       Rayleigh coefficients @f$ \alpha @f$ (mass-proportional, 1/s)
 *       and @f$ \beta @f$ (stiffness-proportional, s) producing a
 *       per-mode rate @f$ d_i = (\alpha + \beta\,\omega_i^2) / 2 @f$.
 */
struct IsotropicMaterial {
    double youngs_modulus;  ///< @f$E@f$, Young's modulus, in pascals (Pa); @f$> 0@f$.
    double poisson_ratio;   ///< @f$\nu@f$, Poisson's ratio (dimensionless);
                            ///< physical range @f$-1 < \nu < 1/2@f$.
    double density;         ///< @f$\rho@f$, mass density (kg / m^3); @f$> 0@f$.
};

}  // namespace chladni
