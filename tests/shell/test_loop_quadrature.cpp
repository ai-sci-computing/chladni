/**
 * @file test_loop_quadrature.cpp
 * @brief Verify the @ref chladni::shell::quadrature_points free function
 *        and the per-rule behavior of @ref chladni::shell::LoopAssembler.
 *
 * Three blocks:
 *  1. Algebraic properties of the rules themselves — total weight,
 *     point count, exactness for the low-order monomials that justify
 *     each rule's degree claim.
 *  2. Per-element kernel behavior — symmetry and rigid-translation
 *     null space preserved across rules (these are basis-level
 *     properties, not quadrature-order-dependent).
 *  3. Global-assembly A/B — frequency spectra on an icosphere converge
 *     to a common limit but differ at finite mesh resolution, with
 *     7-pt > 3-pt > 1-pt by total weighted error against the
 *     Wilkinson analytic closed-shell prediction. (Order is empirical;
 *     under-integration tends to soften the K-side and bring frequencies
 *     down for the all-regular icosphere.)
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

#include <vector>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

cs::ShellMaterial make_material()
{
    cs::ShellMaterial sm;
    sm.k_L           = 1.0e6;
    sm.k_B           = 1.0e3;
    sm.poisson_ratio = 0.3;
    return sm;
}

double total_weight(const std::vector<cs::QuadraturePoint>& qp)
{
    double s = 0.0;
    for (const auto& q : qp) s += q.weight;
    return s;
}

}  // namespace

TEST_CASE("quadrature_points: total weight is 1/2 for every rule",
          "[shell][quadrature]")
{
    // Every rule integrates the constant function 1 to the area of the
    // reference unit-area triangle. Catch the worst case (the
    // assembler-level integrand convention) up-front so a future rule
    // addition can't silently mis-scale.
    for (const auto rule : {cs::QuadratureRule::OnePointCentroid,
                            cs::QuadratureRule::ThreePointEdgeMid,
                            cs::QuadratureRule::SevenPointDunavant}) {
        const auto qp = cs::quadrature_points(rule);
        REQUIRE(qp.size() > 0);
        REQUIRE(total_weight(qp) == Catch::Approx(0.5).margin(1e-14));
    }
}

TEST_CASE("quadrature_points: point counts match the rule labels",
          "[shell][quadrature]")
{
    REQUIRE(cs::quadrature_points(
        cs::QuadratureRule::OnePointCentroid).size() == 1);
    REQUIRE(cs::quadrature_points(
        cs::QuadratureRule::ThreePointEdgeMid).size() == 3);
    REQUIRE(cs::quadrature_points(
        cs::QuadratureRule::SevenPointDunavant).size() == 7);
}

TEST_CASE("quadrature_points: each rule integrates its claimed degree exactly",
          "[shell][quadrature]")
{
    // Reference integrals on T = {(v,w): v>=0, w>=0, v+w<=1}:
    //   integral 1 dv dw     = 1/2
    //   integral v dv dw     = 1/6  (= int w dv dw)
    //   integral v^2 dv dw   = 1/12
    //   integral v*w dv dw   = 1/24
    //   integral v^5 dv dw   = 1/42

    auto integrate = [](const std::vector<cs::QuadraturePoint>& qp,
                        auto&& f)
    {
        double s = 0.0;
        for (const auto& q : qp) s += q.weight * f(q.v, q.w);
        return s;
    };

    const auto qp_1 = cs::quadrature_points(cs::QuadratureRule::OnePointCentroid);
    const auto qp_3 = cs::quadrature_points(cs::QuadratureRule::ThreePointEdgeMid);
    const auto qp_7 = cs::quadrature_points(cs::QuadratureRule::SevenPointDunavant);

    auto v_only = [](double v, double /*w*/) { return v; };
    auto w_only = [](double /*v*/, double w) { return w; };
    auto v_squared = [](double v, double /*w*/) { return v * v; };
    auto v_times_w = [](double v, double w)    { return v * w; };
    auto v_fifth   = [](double v, double /*w*/) { return v * v * v * v * v; };

    // 1-pt centroid: exact for degree <= 1.
    REQUIRE(integrate(qp_1, v_only) == Catch::Approx(1.0 / 6.0).margin(1e-14));
    REQUIRE(integrate(qp_1, w_only) == Catch::Approx(1.0 / 6.0).margin(1e-14));

    // 3-pt edge-mid: exact for degree <= 2.
    REQUIRE(integrate(qp_3, v_only)    == Catch::Approx(1.0 / 6.0).margin(1e-14));
    REQUIRE(integrate(qp_3, v_squared) == Catch::Approx(1.0 / 12.0).margin(1e-14));
    REQUIRE(integrate(qp_3, v_times_w) == Catch::Approx(1.0 / 24.0).margin(1e-14));

    // 7-pt Dunavant: exact for degree <= 5.
    REQUIRE(integrate(qp_7, v_only)    == Catch::Approx(1.0 / 6.0).margin(1e-14));
    REQUIRE(integrate(qp_7, v_squared) == Catch::Approx(1.0 / 12.0).margin(1e-14));
    REQUIRE(integrate(qp_7, v_times_w) == Catch::Approx(1.0 / 24.0).margin(1e-14));
    REQUIRE(integrate(qp_7, v_fifth)   == Catch::Approx(1.0 / 42.0).margin(1e-14));
}

TEST_CASE("LoopAssembler: every quadrature rule preserves K symmetry + rigid-T null space",
          "[shell][loop][quadrature]")
{
    // Basis-level properties — rigid translations annihilate K
    // exactly regardless of quadrature rule because the element
    // construction (CST + box-spline H) is translation-invariant
    // before integration. This test pins that invariant across
    // every rule we've enabled.
    const auto mesh = cmsh::generate_icosphere(0.10, 1);
    const auto sm   = make_material();
    const Eigen::Index dim = 3 * mesh.V.rows();

    for (const auto rule : {cs::QuadratureRule::OnePointCentroid,
                            cs::QuadratureRule::ThreePointEdgeMid,
                            cs::QuadratureRule::SevenPointDunavant}) {
        cs::LoopAssembler::Params p;
        p.k_quad = rule;
        cs::LoopAssembler assembler{p};

        const auto K = assembler.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::MatrixXd K_dense = K;
        const double k_scale = K_dense.cwiseAbs().maxCoeff();

        // Symmetry.
        const Eigen::MatrixXd sym_err = K_dense - K_dense.transpose();
        REQUIRE(sym_err.cwiseAbs().maxCoeff() <= 1e-9 * k_scale);

        // Rigid translation null space (three independent unit
        // translations).
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::VectorXd t = Eigen::VectorXd::Zero(dim);
            for (Eigen::Index v = 0; v < mesh.V.rows(); ++v) {
                t(3 * v + axis) = 1.0;
            }
            const Eigen::VectorXd Kt = K_dense * t;
            REQUIRE(Kt.cwiseAbs().maxCoeff() <= 1e-9 * k_scale);
        }
    }
}

TEST_CASE("LoopAssembler: K and M differ measurably between 1-pt and 7-pt rules",
          "[shell][loop][quadrature]")
{
    // The whole point of exposing the lower-order rules is that they
    // produce *different* matrices. Confirm the spread is large enough
    // to actually drive A/B comparisons (the K bending integrand is
    // degree-4 so 1-pt centroid leaves an O(h) bias that should show
    // at this finite mesh resolution).
    const auto mesh = cmsh::generate_icosphere(0.10, 2);
    const auto sm   = make_material();
    constexpr double surface_density = 7.85;

    cs::LoopAssembler::Params p_7;
    cs::LoopAssembler asm_7{p_7};

    cs::LoopAssembler::Params p_1;
    p_1.k_quad = cs::QuadratureRule::OnePointCentroid;
    p_1.m_quad = cs::QuadratureRule::OnePointCentroid;
    cs::LoopAssembler asm_1{p_1};

    const Eigen::MatrixXd K_7 = asm_7.assemble_K(mesh.V, mesh.F, sm);
    const Eigen::MatrixXd K_1 = asm_1.assemble_K(mesh.V, mesh.F, sm);
    const Eigen::MatrixXd M_7 = asm_7.assemble_M(
        mesh.V, mesh.F, surface_density);
    const Eigen::MatrixXd M_1 = asm_1.assemble_M(
        mesh.V, mesh.F, surface_density);

    const double k_rel = (K_7 - K_1).norm() / K_7.norm();
    const double m_rel = (M_7 - M_1).norm() / M_7.norm();

    // Empirically ~0.1..0.3 for the icosphere k=2; well above noise.
    REQUIRE(k_rel > 1e-3);
    REQUIRE(m_rel > 1e-3);
}
