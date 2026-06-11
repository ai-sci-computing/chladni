/**
 * @file test_loop_element_K.cpp
 * @brief Per-element stiffness K for regular Loop patches: invariants.
 *
 * Tests three sanity invariants on the hex12 flat patch:
 *
 * 1. **Symmetry**. K is built as @f$M^\top H M + B^\top H B@f$ with H
 *    symmetric, so K must be exactly symmetric within floating-point
 *    round-off.
 *
 * 2. **Rigid-translation invariance**. A uniform translation of the
 *    12 control points (the same 3D vector applied to every patch DOF)
 *    is a rigid-body mode and must be annihilated by K. We test all
 *    three coordinate translations independently.
 *
 * 3. **Positive semi-definiteness**. All eigenvalues of K must be
 *    @f$\geq 0@f$ within numerical tolerance. The flat patch admits
 *    at least 6 zero eigenvalues (3 translations + 3 infinitesimal
 *    rotations), so the smallest 6 eigenvalues hover near zero and
 *    the rest are strictly positive.
 *
 * 4. **Pure-bending limit**. Setting @c k_L = 0 (membrane disabled),
 *    K is built only from the bending block. K is still symmetric,
 *    PSD, and translation-invariant.
 *
 * 5. **Pure-membrane limit**. Setting @c k_B = 0, similarly.
 */

#include <chladni/material.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace {

constexpr double kSqrt3Half = 0.8660254037844386;

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

// Steel-like isotropic material at a 1 mm thickness. The specific
// numbers are not critical for invariant tests but are chosen to be
// physically reasonable (close to the analytical reference cases).
chladni::shell::ShellMaterial make_material(double k_L_scale = 1.0,
                                            double k_B_scale = 1.0)
{
    chladni::IsotropicMaterial mat;
    mat.youngs_modulus = 2.0e11;  // Pa
    mat.poisson_ratio  = 0.3;
    mat.density        = 7850.0;
    auto sm = chladni::shell::shell_material_from_isotropic(mat, 1.0e-3);
    sm.k_L *= k_L_scale;
    sm.k_B *= k_B_scale;
    return sm;
}

// Canonical DOFs + V for the hex12 mesh's central triangle (regular
// patch). The element_stiffness_regular interface evaluates the
// patch internally at 7 Dunavant quadrature points, so the test only
// needs to supply the patch's vertex stencil and the V matrix.
struct TestPatchInputs {
    std::array<Eigen::Index, 12> dofs;
    Eigen::MatrixXd              V;
};
TestPatchInputs make_test_patch()
{
    const auto m = make_hex12_mesh();
    const auto patches =
        chladni::shell::loop::build_patch_stencils(m.V, m.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], m.F);
    TestPatchInputs out;
    out.dofs = dofs;
    out.V    = m.V;
    return out;
}

Eigen::Matrix<double, 36, 1> uniform_translation(int axis)
{
    Eigen::Matrix<double, 36, 1> u = Eigen::Matrix<double, 36, 1>::Zero();
    for (int i = 0; i < 12; ++i) {
        u(3 * i + axis) = 1.0;
    }
    return u;
}

}  // namespace

TEST_CASE("element_stiffness_regular: K is symmetric",
          "[shell][loop][K][symmetry]")
{
    const auto p   = make_test_patch();
    const auto mat = make_material();
    const auto K = chladni::shell::loop::element_stiffness_regular(p.dofs, p.V, mat);

    const Eigen::Matrix<double, 36, 36> diff = K - K.transpose();
    REQUIRE(diff.cwiseAbs().maxCoeff() < 1e-6 * K.cwiseAbs().maxCoeff());
}

TEST_CASE("element_stiffness_regular: rigid translation has zero stiffness",
          "[shell][loop][K][translation]")
{
    const auto p   = make_test_patch();
    const auto mat = make_material();
    const auto K = chladni::shell::loop::element_stiffness_regular(p.dofs, p.V, mat);

    const double K_max = K.cwiseAbs().maxCoeff();
    for (int axis = 0; axis < 3; ++axis) {
        CAPTURE(axis);
        const Eigen::Matrix<double, 36, 1> u_t = uniform_translation(axis);
        const Eigen::Matrix<double, 36, 1> Ku  = K * u_t;
        REQUIRE(Ku.cwiseAbs().maxCoeff() < 1e-6 * K_max);
    }
}

TEST_CASE("element_stiffness_regular: K is positive semi-definite",
          "[shell][loop][K][psd]")
{
    const auto p   = make_test_patch();
    const auto mat = make_material();
    const auto K = chladni::shell::loop::element_stiffness_regular(p.dofs, p.V, mat);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 36, 36>> es(K);
    REQUIRE(es.info() == Eigen::Success);
    const Eigen::Matrix<double, 36, 1> ev = es.eigenvalues();
    const double min_ev = ev.minCoeff();
    const double max_ev = ev.maxCoeff();
    // Allow a tiny relative tolerance for the rigid-body modes.
    REQUIRE(min_ev > -1e-6 * max_ev);
}

TEST_CASE("element_stiffness_regular: pure-membrane limit (k_B = 0) is symmetric+PSD",
          "[shell][loop][K][limit][membrane]")
{
    const auto p   = make_test_patch();
    const auto mat = make_material(/*k_L_scale=*/ 1.0, /*k_B_scale=*/ 0.0);
    const auto K = chladni::shell::loop::element_stiffness_regular(p.dofs, p.V, mat);

    REQUIRE((K - K.transpose()).cwiseAbs().maxCoeff()
            < 1e-6 * K.cwiseAbs().maxCoeff());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 36, 36>> es(K);
    REQUIRE(es.eigenvalues().minCoeff() > -1e-6 * es.eigenvalues().maxCoeff());

    // Translation should still be in the kernel.
    const auto K_max = K.cwiseAbs().maxCoeff();
    for (int axis = 0; axis < 3; ++axis) {
        const auto u_t = uniform_translation(axis);
        REQUIRE((K * u_t).cwiseAbs().maxCoeff() < 1e-6 * K_max);
    }
}

TEST_CASE("element_stiffness_regular: pure-bending limit (k_L = 0) is symmetric+PSD",
          "[shell][loop][K][limit][bending]")
{
    const auto p   = make_test_patch();
    const auto mat = make_material(/*k_L_scale=*/ 0.0, /*k_B_scale=*/ 1.0);
    const auto K = chladni::shell::loop::element_stiffness_regular(p.dofs, p.V, mat);

    REQUIRE((K - K.transpose()).cwiseAbs().maxCoeff()
            < 1e-6 * K.cwiseAbs().maxCoeff());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 36, 36>> es(K);
    REQUIRE(es.eigenvalues().minCoeff() > -1e-6 * es.eigenvalues().maxCoeff());

    const auto K_max = K.cwiseAbs().maxCoeff();
    for (int axis = 0; axis < 3; ++axis) {
        const auto u_t = uniform_translation(axis);
        REQUIRE((K * u_t).cwiseAbs().maxCoeff() < 1e-6 * K_max);
    }
}
