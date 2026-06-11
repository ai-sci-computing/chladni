/**
 * @file test_stiffness.cpp
 * @brief Invariants of the FD-assembled discrete-shells stiffness matrix.
 *
 * For modal analysis at small displacements the relevant K is the Hessian
 * of @ref chladni::shell::shell_energy at the rest configuration. Two
 * algebraic invariants must hold regardless of the mesh, material, or
 * thickness:
 *
 *   - K is symmetric (W is C^infty, so its Hessian is symmetric; FD
 *     introduces tiny asymmetry which the assembler symmetrizes).
 *   - Rigid body translations are zero modes: K * (e_alpha (X) 1_n) ~ 0
 *     for alpha in {x, y, z}.
 *
 * A weaker rigid-rotation invariance also holds for u_i = omega x x_i,
 * but only at first order in displacement; with finite epsilon and
 * finite mesh the residual is dominated by O(eps^2) rather than zero.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <filesystem>

namespace {

namespace fs = std::filesystem;

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9, .poisson_ratio = 0.30, .density = 7850.0};
}

}  // namespace

TEST_CASE("stiffness K on cylinder.obj is symmetric within FD noise",
          "[shell][stiffness][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd  = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    const auto K = chladni::shell::assemble_stiffness_at_rest_fd(
        mesh.V, mesh.F, edges, rd, mat);

    REQUIRE(K.rows() == 3 * mesh.V.rows());
    REQUIRE(K.cols() == 3 * mesh.V.rows());

    const Eigen::SparseMatrix<double> Kt = K.transpose();
    const Eigen::SparseMatrix<double> diff = K - Kt;
    const double Kabs = K.coeffs().cwiseAbs().maxCoeff();
    const double D    = diff.coeffs().cwiseAbs().maxCoeff();
    INFO("max |K|=" << Kabs << "  max |K - K^T|=" << D);
    REQUIRE(D < 1e-9 * Kabs);
}

TEST_CASE("analytic and FD stiffness at rest agree on cylinder.obj",
          "[shell][stiffness][analytic][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd  = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    const auto K_fd  = chladni::shell::assemble_stiffness_at_rest_fd(
        mesh.V, mesh.F, edges, rd, mat);
    const auto K_an  = chladni::shell::assemble_stiffness_at_rest_analytic(
        mesh.V, mesh.F, edges, rd, mat);

    REQUIRE(K_an.rows() == K_fd.rows());
    REQUIRE(K_an.cols() == K_fd.cols());

    const Eigen::MatrixXd K_fd_d = Eigen::MatrixXd(K_fd);
    const Eigen::MatrixXd K_an_d = Eigen::MatrixXd(K_an);

    const double max_abs_K   = K_fd_d.cwiseAbs().maxCoeff();
    const double max_abs_dif = (K_fd_d - K_an_d).cwiseAbs().maxCoeff();
    INFO("max |K_fd| = " << max_abs_K);
    INFO("max |K_fd - K_an| = " << max_abs_dif);
    INFO("relative diff = " << max_abs_dif / max_abs_K);

    // FD truncation error on cylinder.obj at eps=1e-6 is ~1e-4 relative
    // for K_membrane and a bit looser for K_bending. The analytic
    // expression is the exact small-displacement Hessian, so the
    // disagreement is bounded above by the FD truncation alone.
    REQUIRE(max_abs_dif < 1e-3 * max_abs_K);
}

TEST_CASE("stiffness K on cylinder.obj annihilates rigid translations",
          "[shell][stiffness][cylinder][rigid_body]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd  = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    const auto mat = chladni::shell::shell_material_from_isotropic(steel(), 1.0e-3);

    const auto K = chladni::shell::assemble_stiffness_at_rest_fd(
        mesh.V, mesh.F, edges, rd, mat);

    const Eigen::Index n = mesh.V.rows();
    const Eigen::Index dim = 3 * n;
    const double scale = K.coeffs().cwiseAbs().maxCoeff();
    REQUIRE(scale > 0);

    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(dim);
        for (Eigen::Index v = 0; v < n; ++v) {
            u(3 * v + axis) = 1.0;
        }
        const Eigen::VectorXd Ku = K * u;
        const double residual = Ku.cwiseAbs().maxCoeff();
        INFO("axis=" << axis << "  max |K u_t|=" << residual
             << "  scale=" << scale << "  ratio=" << residual / scale);
        // Rigid body invariance is exact mathematically; numerical floor
        // here is set by the FD step relative to scale.
        REQUIRE(residual < 1e-6 * scale);
    }
}
