/**
 * @file test_loop_evaluate.cpp
 * @brief evaluate_patch_regular: flat-patch zero-curvature and affine
 *        reproduction tests for the regular Loop limit surface.
 *
 * The regular Loop subdivision basis (Cirak-Ortiz Eq. 75) reproduces
 * affine functions exactly when the 12 control points are placed at
 * the standard hexagonal grid positions in canonical Fig. 9 slot
 * order. Tests:
 *
 *  1. **Position is in-plane and matches the affine formula.** Setting
 *     all 12 control z-coordinates to zero, position(v, w) must lie at
 *     z = 0. The (x, y) components must agree with the affine map
 *     induced by the central triangle's three corners.
 *
 *  2. **Parametric Hessian vanishes on a flat patch.** Affine
 *     reproduction implies @f$\partial^2 x / \partial v^2 = 0@f$ etc.
 *     for every spatial component.
 *
 *  3. **Surface normal is the patch normal.** All control points lie
 *     in z = 0, so @f$a_3 = (0, 0, \pm 1)@f$.
 *
 *  4. **Affine reproduction in 3D.** Apply an arbitrary affine map
 *     @f$P_I \mapsto a + L \cdot P_I@f$ to all 12 control points
 *     (lifting the patch into 3D in a non-axis-aligned way) and
 *     verify the limit position transforms accordingly. This is a
 *     sanity check that the basis is genuinely 3-equivariant.
 */

#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <vector>

namespace {

constexpr double kSqrt3Half = 0.8660254037844386;  // sqrt(3)/2

// 12-vertex hex mesh in the xy-plane. Same as test_loop_canonical's
// fixture; duplicated here to keep the test files independent.
struct Hex12Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};
Hex12Mesh make_hex12_mesh()
{
    const double h = kSqrt3Half;
    Hex12Mesh m;
    m.V.resize(12, 3);
    m.V.row(0)  << -0.5,         h,   0.0;
    m.V.row(1)  <<  0.0,    2.0 * h,  0.0;
    m.V.row(2)  <<  0.0,    0.0,      0.0;
    m.V.row(3)  <<  0.5,         h,   0.0;
    m.V.row(4)  <<  1.0,    2.0 * h,  0.0;
    m.V.row(5)  <<  0.5,        -h,   0.0;
    m.V.row(6)  <<  1.0,    0.0,      0.0;
    m.V.row(7)  <<  1.5,         h,   0.0;
    m.V.row(8)  <<  2.0,    2.0 * h,  0.0;
    m.V.row(9)  <<  1.5,        -h,   0.0;
    m.V.row(10) <<  2.0,    0.0,      0.0;
    m.V.row(11) <<  2.5,         h,   0.0;
    m.F.resize(13, 3);
    m.F.row(0)  << 3, 6, 7;
    m.F.row(1)  << 3, 7, 4;
    m.F.row(2)  << 3, 4, 1;
    m.F.row(3)  << 3, 1, 0;
    m.F.row(4)  << 3, 0, 2;
    m.F.row(5)  << 3, 2, 6;
    m.F.row(6)  << 6, 2, 5;
    m.F.row(7)  << 6, 5, 9;
    m.F.row(8)  << 6, 9, 10;
    m.F.row(9)  << 6, 10, 7;
    m.F.row(10) << 7, 10, 11;
    m.F.row(11) << 7, 11, 8;
    m.F.row(12) << 7, 8,  4;
    return m;
}

// Affine map (x, y) -> position induced by the central triangle's
// 3 corners (slots 3, 6, 7 of the hex12 mesh, i.e. vertices 4, 7, 8
// in 1-indexed Fig. 9 notation):
//   - corner 0 (slot 3) at (0.5, h):      maps to (u=1, v=0, w=0)
//   - corner 1 (slot 6) at (1.0, 0.0):    maps to (u=0, v=1, w=0)
//   - corner 2 (slot 7) at (1.5, h):      maps to (u=0, v=0, w=1)
// position(v, w) inside the central triangle = (1-v-w)*P0 + v*P1 + w*P2.
Eigen::Vector2d expected_central_position(double v, double w)
{
    const double h = kSqrt3Half;
    const double u = 1.0 - v - w;
    return {u * 0.5 + v * 1.0 + w * 1.5,
            u * h   + v * 0.0 + w * h};
}

const std::vector<std::pair<double, double>> centroid_and_friends = {
    {1.0 / 3.0, 1.0 / 3.0},   // centroid (Cirak-Ortiz one-point quadrature)
    {0.5,        0.25      },
    {0.25,       0.25      },
    {0.1,        0.6       },
    {0.7,        0.2       },
};

}  // namespace

TEST_CASE("evaluate_patch_regular: flat patch position lies in z = 0",
          "[shell][loop][evaluate][flat]")
{
    const auto mesh = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(mesh.V, mesh.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], mesh.F);

    for (const auto& [v, w] : centroid_and_friends) {
        CAPTURE(v, w);
        const auto pe =
            chladni::shell::loop::evaluate_patch_regular(dofs, mesh.V, v, w);
        REQUIRE(pe.position.z() == Catch::Approx(0.0).margin(1e-13));

        const Eigen::Vector2d expected = expected_central_position(v, w);
        REQUIRE(pe.position.x() == Catch::Approx(expected.x()).margin(1e-13));
        REQUIRE(pe.position.y() == Catch::Approx(expected.y()).margin(1e-13));
    }
}

TEST_CASE("evaluate_patch_regular: flat patch parametric Hessian is zero",
          "[shell][loop][evaluate][flat][curvature]")
{
    const auto mesh = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(mesh.V, mesh.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], mesh.F);

    for (const auto& [v, w] : centroid_and_friends) {
        CAPTURE(v, w);
        const auto pe =
            chladni::shell::loop::evaluate_patch_regular(dofs, mesh.V, v, w);
        for (int col = 0; col < 3; ++col) {
            CAPTURE(col);
            REQUIRE(pe.second_derivs(0, col)
                    == Catch::Approx(0.0).margin(1e-13));
            REQUIRE(pe.second_derivs(1, col)
                    == Catch::Approx(0.0).margin(1e-13));
            REQUIRE(pe.second_derivs(2, col)
                    == Catch::Approx(0.0).margin(1e-13));
        }
    }
}

TEST_CASE("evaluate_patch_regular: flat patch normal is +/- e_z",
          "[shell][loop][evaluate][flat][normal]")
{
    const auto mesh = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(mesh.V, mesh.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], mesh.F);

    for (const auto& [v, w] : centroid_and_friends) {
        CAPTURE(v, w);
        const auto pe =
            chladni::shell::loop::evaluate_patch_regular(dofs, mesh.V, v, w);
        REQUIRE(pe.normal.x() == Catch::Approx(0.0).margin(1e-13));
        REQUIRE(pe.normal.y() == Catch::Approx(0.0).margin(1e-13));
        REQUIRE(std::abs(pe.normal.z()) == Catch::Approx(1.0).margin(1e-13));
    }
}

TEST_CASE("evaluate_patch_regular: tangent basis matches affine map",
          "[shell][loop][evaluate][flat][tangent]")
{
    // For the flat patch, position is affine in (v, w) with the
    // explicit gradient
    //   d(position)/d(v) = P1 - P0 = (0.5, -h, 0)
    //   d(position)/d(w) = P2 - P0 = (1.0, 0,  0)
    const auto mesh = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(mesh.V, mesh.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], mesh.F);

    const Eigen::Vector3d expected_v(0.5, -kSqrt3Half, 0.0);
    const Eigen::Vector3d expected_w(1.0, 0.0,         0.0);

    for (const auto& [v, w] : centroid_and_friends) {
        CAPTURE(v, w);
        const auto pe =
            chladni::shell::loop::evaluate_patch_regular(dofs, mesh.V, v, w);
        for (int k = 0; k < 3; ++k) {
            REQUIRE(pe.cov_basis(k, 0)
                    == Catch::Approx(expected_v(k)).margin(1e-13));
            REQUIRE(pe.cov_basis(k, 1)
                    == Catch::Approx(expected_w(k)).margin(1e-13));
        }
    }
}

TEST_CASE("evaluate_patch_regular: 3D affine equivariance",
          "[shell][loop][evaluate][affine]")
{
    // Apply an arbitrary affine map P -> a + L * P to every control
    // point, then verify position(v, w) transforms the same way. This
    // confirms the basis is fully 3-equivariant (not just z = 0
    // preserving).
    const auto mesh = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(mesh.V, mesh.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], mesh.F);

    const Eigen::Vector3d a(1.5, -2.0, 3.5);
    Eigen::Matrix3d L;
    L <<  1.2,  0.4, -0.7,
         -0.5,  1.1,  0.3,
          0.2, -0.6,  0.9;

    Eigen::MatrixXd V_xform(12, 3);
    for (int i = 0; i < 12; ++i) {
        const Eigen::Vector3d p = mesh.V.row(i).transpose();
        V_xform.row(i) = (a + L * p).transpose();
    }

    for (const auto& [v, w] : centroid_and_friends) {
        CAPTURE(v, w);
        const auto pe_orig = chladni::shell::loop::evaluate_patch_regular(
            dofs, mesh.V, v, w);
        const auto pe_xform = chladni::shell::loop::evaluate_patch_regular(
            dofs, V_xform, v, w);
        const Eigen::Vector3d expected = a + L * pe_orig.position;
        for (int k = 0; k < 3; ++k) {
            REQUIRE(pe_xform.position(k)
                    == Catch::Approx(expected(k)).margin(1e-12));
        }
    }
}
