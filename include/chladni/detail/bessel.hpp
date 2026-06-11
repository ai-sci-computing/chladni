#pragma once

/**
 * @file bessel.hpp
 * @brief Integer-order cylindrical Bessel functions for the analytical
 *        references.
 *
 * Several closed-form thin-plate references in
 * @ref chladni::analytical (annular plate, free-edge circular plate)
 * need the Bessel functions @f$J_n@f$, @f$Y_n@f$, @f$I_n@f$, @f$K_n@f$
 * of integer order @f$n \ge 0@f$ at moderate real arguments
 * @f$x \in (0, 30]@f$. Apple Clang's libc++ does not yet ship
 * @c std::cyl_bessel_j et al. (added in mainline LLVM 18+, still missing
 * in Apple Clang 21 as of macOS 15), so we implement the eight base
 * cases @f$Z_0(x)@f$ and @f$Z_1(x)@f$ directly via the
 * Abramowitz-Stegun rational approximations and recurse forward in @c n.
 *
 * @note This is a project-private implementation header
 *       (note the @c chladni::detail namespace). It is not part of the
 *       public API; behaviour outside the documented argument and order
 *       ranges is unspecified.
 *
 * @section accuracy Accuracy
 * The base-case approximations from @cite abramowitz_stegun are accurate
 * to roughly @f$10^{-7}@f$ across their respective argument ranges; that
 * is amply sufficient to find determinant zeros to four-significant-figure
 * agreement with published tables.
 */

namespace chladni::detail {

/**
 * @brief Bessel function of the first kind, @f$J_n(x)@f$, for integer
 *        @f$n \ge 0@f$ and real @f$x > 0@f$.
 *
 * Computed from the @f$J_0@f$ and @f$J_1@f$ Abramowitz-Stegun rational
 * approximations and forward recurrence
 * @f$ J_{n+1}(x) = (2n/x)\,J_n(x) - J_{n-1}(x) @f$.
 * Forward recurrence is well-conditioned in the regime @f$ n \le x @f$;
 * it loses accuracy for @f$ n \gg x @f$ but the analytical references
 * built on top of this header only ever need @f$ n \le 5 @f$.
 *
 * @warning Forward recurrence is unstable for @f$ x < n @f$ and there is
 *          no internal guard. Current callers stay safe only because the
 *          plate solvers gate their scans to the stable @f$ n \le x @f$
 *          regime; a new caller passing @f$ x \ll n @f$ gets a silently
 *          inaccurate result.
 *
 * @param n  Order, @f$\ge 0@f$.
 * @param x  Argument, @f$ > 0@f$.
 * @return   @f$J_n(x)@f$.
 */
double bessel_J(int n, double x);

/**
 * @brief Bessel function of the second kind (Neumann), @f$Y_n(x)@f$,
 *        for integer @f$n \ge 0@f$ and real @f$x > 0@f$.
 *
 * Computed from the @f$Y_0@f$ and @f$Y_1@f$ Abramowitz-Stegun rational
 * approximations and forward recurrence
 * @f$ Y_{n+1}(x) = (2n/x)\,Y_n(x) - Y_{n-1}(x) @f$.
 * @f$Y_n(x)@f$ has a logarithmic singularity at the origin; this
 * routine returns large negative values as @f$x \to 0^+@f$ but does
 * not guard against @f$x \le 0@f$.
 */
double bessel_Y(int n, double x);

/**
 * @brief Modified Bessel function of the first kind, @f$I_n(x)@f$,
 *        for integer @f$n \ge 0@f$ and real @f$x > 0@f$.
 *
 * Forward recurrence
 * @f$ I_{n+1}(x) = I_{n-1}(x) - (2n/x)\,I_n(x) @f$
 * is conditionally stable: it suffices for the small @f$n@f$ used here
 * but is not safe for @f$ n \gg x @f$.
 *
 * @warning Unstable for @f$ x < n @f$ with no internal guard — see the
 *          @ref bessel_J warning; the same gating keeps current callers
 *          in the safe regime.
 */
double bessel_I(int n, double x);

/**
 * @brief Modified Bessel function of the second kind, @f$K_n(x)@f$,
 *        for integer @f$n \ge 0@f$ and real @f$x > 0@f$.
 *
 * Forward recurrence
 * @f$ K_{n+1}(x) = K_{n-1}(x) + (2n/x)\,K_n(x) @f$
 * is unconditionally stable for @f$x > 0@f$ (all terms positive).
 */
double bessel_K(int n, double x);

/**
 * @brief First derivative @f$ J_n'(x) @f$ via the standard recurrence
 *        @f$ J_n'(x) = J_{n-1}(x) - (n/x) J_n(x) @f$,
 *        with the convention @f$ J_0'(x) = -J_1(x) @f$.
 */
double bessel_J_prime(int n, double x);

/**
 * @brief First derivative @f$ Y_n'(x) @f$ via
 *        @f$ Y_n'(x) = Y_{n-1}(x) - (n/x) Y_n(x) @f$,
 *        with @f$ Y_0'(x) = -Y_1(x) @f$.
 */
double bessel_Y_prime(int n, double x);

/**
 * @brief First derivative @f$ I_n'(x) @f$ via
 *        @f$ I_n'(x) = I_{n-1}(x) - (n/x) I_n(x) @f$,
 *        with @f$ I_0'(x) = I_1(x) @f$.
 *
 * Note the sign difference vs. @f$Y_n'@f$: the modified-Bessel
 * recurrence has @f$+I_{n+1}@f$ on the other side.
 */
double bessel_I_prime(int n, double x);

/**
 * @brief First derivative @f$ K_n'(x) @f$ via
 *        @f$ K_n'(x) = -K_{n-1}(x) - (n/x) K_n(x) @f$,
 *        with @f$ K_0'(x) = -K_1(x) @f$.
 *
 * @f$K_n@f$ is monotonically decreasing in @f$x@f$ so all derivatives
 * are negative.
 */
double bessel_K_prime(int n, double x);

}  // namespace chladni::detail
