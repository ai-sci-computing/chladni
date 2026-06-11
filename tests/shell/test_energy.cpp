/**
 * @file test_energy.cpp
 * @brief Invariants of the discrete-shells energy chladni::shell::shell_energy.
 *
 * Tests on cylinder.obj with a steel ShellMaterial:
 *   - W(rest) ~ 0
 *   - W(rest + uniform translation) = 0  (rigid body invariance)
 *   - W(rest + tiny in-plane stretch) > 0 (membrane responds)
 *   - W(rest + tiny normal pinch) > 0     (bending responds)
 *   - membrane-only and bending-only energies are independent toggles.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

#include <filesystem>

namespace {

namespace fs = std::filesystem;

chladni::mesh::TriMesh load_cylinder()
{
    return chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
}

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9, .poisson_ratio = 0.30, .density = 7850.0};
}

}  // namespace

TEST_CASE("shell_energy: zero at rest, zero under translation", "[shell][energy]")
{
    const auto mesh = load_cylinder();
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    SECTION("W(rest) is essentially zero") {
        const double W = chladni::shell::shell_energy(
            mesh.V, mesh.V, mesh.F, edges, rd, mat);
        // Rest dihedrals are computed from the same vertices, so
        // theta - theta_bar = 0 and L - L_bar = 0 to machine precision.
        REQUIRE(W < 1e-18);
    }

    SECTION("W is invariant under rigid translation") {
        Eigen::MatrixXd Vt = mesh.V;
        for (Eigen::Index i = 0; i < Vt.rows(); ++i) {
            Vt(i, 0) += 1.7;   // arbitrary translation
            Vt(i, 1) += -3.4;
            Vt(i, 2) += 0.5;
        }
        const double W = chladni::shell::shell_energy(
            mesh.V, Vt, mesh.F, edges, rd, mat);
        REQUIRE(W < 1e-12);
    }
}

TEST_CASE("shell_energy: in-plane stretch raises the membrane term",
          "[shell][energy][membrane]")
{
    const auto mesh = load_cylinder();
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    // Uniform axial stretch: scale z by 1.001 (0.1% strain).
    Eigen::MatrixXd Vs = mesh.V;
    Vs.col(2) *= 1.001;

    const double W_with_membrane =
        chladni::shell::shell_energy(mesh.V, Vs, mesh.F, edges, rd, mat);

    // Same deformation with zero membrane stiffness.
    chladni::shell::ShellMaterial bend_only = mat;
    bend_only.k_L = 0.0;
    const double W_bending_only =
        chladni::shell::shell_energy(mesh.V, Vs, mesh.F, edges, rd, bend_only);

    // The membrane contribution must be strictly positive for an in-plane
    // stretch and dominate by orders of magnitude over the bending residue.
    REQUIRE(W_with_membrane > W_bending_only);
    REQUIRE((W_with_membrane - W_bending_only) > 1.0);  // 1 J for the cylinder
}

TEST_CASE("shell_energy: normal pinch raises the bending term",
          "[shell][energy][bending]")
{
    const auto mesh = load_cylinder();
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    // Push every other vertex slightly inward along its xy direction.
    // This bends panels relative to their neighbours without significant
    // edge-length change at first order.
    Eigen::MatrixXd Vp = mesh.V;
    for (Eigen::Index i = 0; i < Vp.rows(); i += 2) {
        const double r = std::hypot(Vp(i, 0), Vp(i, 1));
        if (r > 0.0) {
            Vp(i, 0) -= 1.0e-4 * Vp(i, 0) / r;
            Vp(i, 1) -= 1.0e-4 * Vp(i, 1) / r;
        }
    }

    chladni::shell::ShellMaterial mem_only = mat;
    mem_only.k_B = 0.0;

    const double W_full = chladni::shell::shell_energy(
        mesh.V, Vp, mesh.F, edges, rd, mat);
    const double W_membrane_only = chladni::shell::shell_energy(
        mesh.V, Vp, mesh.F, edges, rd, mem_only);

    // Bending term contributes a strictly positive amount.
    REQUIRE(W_full > W_membrane_only);
    REQUIRE((W_full - W_membrane_only) > 0.0);
}

TEST_CASE("shell_material_from_isotropic: matches the standard plate-rigidity formulae",
          "[shell][energy][material]")
{
    // Aluminum
    const chladni::IsotropicMaterial al{
        .youngs_modulus = 69.0e9,
        .poisson_ratio  = 0.33,
        .density        = 2700.0,
    };
    constexpr double h = 1.0e-3;

    const auto mat = chladni::shell::shell_material_from_isotropic(al, h);
    const double D_expected =
        69.0e9 * h * h * h / (12.0 * (1.0 - 0.33 * 0.33));
    const double Eh_expected = 69.0e9 * h / (1.0 - 0.33 * 0.33);

    REQUIRE(mat.k_B == Catch::Approx(D_expected).epsilon(1e-12));
    REQUIRE(mat.k_L == Catch::Approx(Eh_expected).epsilon(1e-12));
    REQUIRE(mat.poisson_ratio == Catch::Approx(0.33).epsilon(1e-12));
}

TEST_CASE("shell_material_from_isotropic: rejects bad inputs",
          "[shell][energy][material][validation]")
{
    const chladni::IsotropicMaterial al{
        .youngs_modulus = 69.0e9,
        .poisson_ratio  = 0.33,
        .density        = 2700.0,
    };
    REQUIRE_THROWS_AS(
        chladni::shell::shell_material_from_isotropic(al, 0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        chladni::shell::shell_material_from_isotropic(
            chladni::IsotropicMaterial{.youngs_modulus = -1.0,
                                       .poisson_ratio = 0.3,
                                       .density = 1.0},
            1.0),
        std::invalid_argument);
}
