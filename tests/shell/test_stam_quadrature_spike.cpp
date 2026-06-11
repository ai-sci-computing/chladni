/**
 * @file test_stam_quadrature_spike.cpp
 * @brief R12 scoped spike: does the single 7-point Dunavant rule on the
 *        whole reference triangle under-integrate the Stam irregular patch?
 *
 * element_stiffness_stam / element_mass_stam place ONE reference-triangle
 * quadrature rule over the whole unit triangle. But the Stam limit basis is
 * piecewise-polynomial across the dyadic tiles Omega_k^n that shrink onto
 * the extraordinary vertex (0,0): a fixed rule's innermost sample sits at
 * tile level ~3 (Dunavant's closest point is (0.10,0.10), sum=0.20), so it
 * MISSES every tile at level >= 4. Their basis-gradient energy is a
 * convergent-but-nonzero geometric tail (the gradient picks up 2^n per tile
 * via the chain rule, damped by the eigenvalue scaling Lambda^{n-1} < 1).
 *
 * This probe integrates a representative basis-gradient energy density
 *   g(v,w) = sum_b |dPhi_b/dv(v,w)|^2
 * two ways: (a) the single 7-point rule, and (b) an exact-per-tile rule that
 * partitions the reference triangle into n_rings dyadic rings (3 tiles each)
 * and applies Dunavant on each tile mapped back to local coords. The
 * deep-subdivision value is the converged ("exact") integral; comparing the
 * single rule to it quantifies the under-integration the review (R12)
 * flagged. Run explicitly: STAM_DIAG=1 ./chladni_tests "[stam_quad_spike]".
 */

#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using chladni::shell::QuadraturePoint;

/// Map a Dunavant point given in a tile's local unit-triangle coords
/// (v_p, w_p) back to the reference triangle (v, w) for dyadic ring n and
/// sub-tile k (k in {1,2,3}), scaling the weight by the tile Jacobian 1/4^n.
/// Inverse of chladni::shell::stam_tile_map.
QuadraturePoint tile_to_reference(const QuadraturePoint& p, int n, int k)
{
    const double inv2n = std::ldexp(1.0, -n);   // 2^-n
    double v = 0.0, w = 0.0;
    if (k == 1) {            // corner tile at v=1
        v = (p.v + 1.0) * inv2n;
        w =  p.w * inv2n;
    } else if (k == 3) {     // corner tile at w=1
        v =  p.v * inv2n;
        w = (p.w + 1.0) * inv2n;
    } else {                 // k == 2, medial tile (flipped)
        v = (1.0 - p.v) * inv2n;
        w = (1.0 - p.w) * inv2n;
    }
    const double jac = std::ldexp(1.0, -2 * n);  // 4^-n
    return QuadraturePoint{v, w, p.weight * jac};
}

/// Build a per-tile subdivided quadrature: each of n_rings dyadic rings
/// contributes its 3 sub-tiles, each carrying the base rule mapped to local
/// coords. The tail below ring n_rings (parametric area (1/2)*4^-n_rings) is
/// dropped — its contribution vanishes geometrically.
std::vector<QuadraturePoint> subdivided_rule(
    chladni::shell::QuadratureRule base, int n_rings)
{
    const auto base_pts = chladni::shell::quadrature_points(base);
    std::vector<QuadraturePoint> out;
    out.reserve(static_cast<std::size_t>(3 * n_rings) * base_pts.size());
    for (int n = 1; n <= n_rings; ++n) {
        for (int k = 1; k <= 3; ++k) {
            for (const auto& p : base_pts) {
                out.push_back(tile_to_reference(p, n, k));
            }
        }
    }
    return out;
}

/// Integrate g(v,w) = sum_b |dPhi_b/dv|^2 over the reference triangle with
/// the given quadrature point set.
double integrate_grad_energy(const chladni::shell::loop::StamEvaluator& ev,
                             const std::vector<QuadraturePoint>& pts)
{
    double acc = 0.0;
    for (const auto& q : pts) {
        const auto grad = chladni::shell::loop::stam_phi_grad(ev, q.v, q.w);
        acc += q.weight * grad.col(0).squaredNorm();
    }
    return acc;
}

}  // namespace

TEST_CASE("R12 spike: Stam single-rule vs exact-per-tile quadrature",
          "[.diag][shell][stam][stam_quad_spike]")
{
    const bool diag = std::getenv("STAM_DIAG") != nullptr;
    using chladni::shell::QuadratureRule;

    for (int N : {3, 5, 6, 7}) {   // valences: 3 (Jordan), 5/7 (icosphere), 6
        const auto ev = chladni::shell::loop::make_stam_evaluator(N);

        const double single = integrate_grad_energy(
            ev, chladni::shell::quadrature_points(
                    QuadratureRule::SevenPointDunavant));

        // Deep subdivision is the converged reference.
        const double exact = integrate_grad_energy(
            ev, subdivided_rule(QuadratureRule::SevenPointDunavant, 20));

        if (diag) {
            std::printf("[stam_quad_spike] N=%d  single7=%.10e  exact=%.10e  "
                        "rel_err=%.3e\n",
                        N, single, exact, std::abs(single - exact) /
                            std::max(1e-300, std::abs(exact)));
            for (int nr : {1, 2, 3, 4, 6, 8, 12}) {
                const double sub = integrate_grad_energy(
                    ev, subdivided_rule(QuadratureRule::SevenPointDunavant, nr));
                std::printf("    n_rings=%2d  val=%.10e  rel_to_exact=%.3e\n",
                            nr, sub, std::abs(sub - exact) /
                                std::max(1e-300, std::abs(exact)));
            }
        }

        // Sanity: the integral is finite and positive, and the subdivided
        // rule converges (deep value is stable to a further doubling).
        const double exact2 = integrate_grad_energy(
            ev, subdivided_rule(QuadratureRule::SevenPointDunavant, 24));
        REQUIRE(std::isfinite(exact));
        REQUIRE(exact > 0.0);
        REQUIRE(std::abs(exact2 - exact) / exact < 1e-6);
    }
}
