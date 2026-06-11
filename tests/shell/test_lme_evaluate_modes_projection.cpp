/**
 * @file test_lme_evaluate_modes_projection.cpp
 * @brief R2: LMEAssembler::evaluate_modes_at_vertices slices the real-vertex
 *        coefficients, and that slice is the EXACT vertex projection because
 *        the composite curved LME basis interpolates at the mesh vertices.
 *
 * The review (R2) asked for an explicit projection u(x_j) = sum_a N_a(x_j) c_a
 * on the assumption that the max-ent basis is not interpolating. Measurement
 * showed it IS interpolating at vertices: a query at vertex x_j projects onto
 * node j's own chart position in every chart that contains j, so each active
 * patch's in-chart solve hits its exact-node match and returns {j: 1}, and the
 * partition of unity sums these to N_a(x_j) = delta_aj. This file guards both
 * the building-block property (evaluate_basis_curved returns a Kronecker
 * delta at a node) and the high-level contract (evaluate_modes_at_vertices
 * preserves a constant field and is finite / correctly shaped).
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <numbers>
#include <vector>

namespace {

namespace cl = chladni::shell::lme;

/// A flat regular hexagon: centre vertex 0 at the origin, six ring vertices.
Eigen::MatrixXd hex_patch_nodes()
{
    Eigen::MatrixXd V(7, 3);
    V.setZero();
    for (int k = 0; k < 6; ++k) {
        const double a = std::numbers::pi_v<double> * k / 3.0;
        V(k + 1, 0) = std::cos(a);
        V(k + 1, 1) = std::sin(a);
    }
    return V;
}

}  // namespace

TEST_CASE("evaluate_basis_curved interpolates at chart nodes (R2 basis guard)",
          "[shell][lme][evaluate_modes][projection]")
{
    const Eigen::MatrixXd nodes = hex_patch_nodes();
    std::vector<int> neighbour_ids = {0, 1, 2, 3, 4, 5, 6};
    const Eigen::VectorXd beta_wpca = Eigen::VectorXd::Ones(7);

    const cl::Patch patch =
        cl::build_patch(/*anchor_id=*/0, nodes, neighbour_ids, beta_wpca);

    std::vector<cl::Patch> patches = {patch};
    const Eigen::MatrixXd  patch_points = nodes.row(0);   // one patch centre
    const Eigen::VectorXd  beta_patches = Eigen::VectorXd::Ones(1);
    const Eigen::VectorXd  beta_lme     = Eigen::VectorXd::Ones(7);

    // Query at every node: the composite basis must return a Kronecker delta
    // (interpolation) — the property that makes the coefficient slice exact.
    for (int b = 0; b < nodes.rows(); ++b) {
        const Eigen::Vector3d y = nodes.row(b).transpose();
        const cl::CurvedBasisWeights w = cl::evaluate_basis_curved(
            patches, patch_points, beta_patches, nodes, beta_lme, y,
            /*tol_shepard=*/1.0e-10, /*r_cut=*/1.0e6,
            /*newton_tol=*/1.0e-10, /*newton_max_iters=*/50);

        double w_at_b = 0.0, w_elsewhere = 0.0;
        for (std::size_t t = 0; t < w.indices.size(); ++t) {
            if (w.indices[t] == b) w_at_b = w.values[t];
            else w_elsewhere += std::abs(w.values[t]);
        }
        INFO("node " << b << ": w_at_b=" << w_at_b
                     << " w_elsewhere=" << w_elsewhere);
        REQUIRE(std::abs(w_at_b - 1.0) < 1e-9);
        REQUIRE(w_elsewhere < 1e-9);
    }
}

TEST_CASE("evaluate_modes_at_vertices: constant field preserved, finite, "
          "shaped (R2 contract)",
          "[shell][lme][evaluate_modes][projection]")
{
    const auto mesh = chladni::mesh::generate_disk_iso(/*radius=*/0.10,
                                                       /*n_boundary=*/20);
    const Eigen::Index n_v = mesh.V.rows();

    const chladni::IsotropicMaterial iso{
        .youngs_modulus = 200.0e9, .poisson_ratio = 0.30, .density = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(iso, 1.0e-3);

    chladni::shell::LMEAssembler assembler{};  // LME, curved, ghost defaults
    const Eigen::Index n_dof =
        assembler.assemble_K(mesh.V, mesh.F, sm).rows();
    REQUIRE(n_dof > 3 * n_v);  // ghost rows were appended
    const Eigen::Index n_ext = n_dof / 3;

    // A constant coefficient field must map to that constant at every vertex
    // (partition of unity), and the output must be finite and 3*n_v x cols.
    Eigen::MatrixXd coeffs = Eigen::MatrixXd::Ones(3 * n_ext, 1);
    const Eigen::MatrixXd out =
        assembler.evaluate_modes_at_vertices(mesh.V, mesh.F, coeffs);
    REQUIRE(out.rows() == 3 * n_v);
    REQUIRE(out.cols() == 1);
    REQUIRE(out.allFinite());
    for (Eigen::Index i = 0; i < out.rows(); ++i) {
        REQUIRE(std::abs(out(i, 0) - 1.0) < 1e-9);
    }
}
