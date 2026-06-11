/**
 * @file test_modes_vs_ss_rect_plate_analytic.cpp
 * @brief Loop FEM vs Leissa: simply-supported rectangular thin plate.
 *
 * Validates the Loop FEM by comparing the lowest few transverse-bending
 * modes of a flat plate against the closed-form Navier (simply-supported
 * on all four edges) result
 * @f[
 *     \omega_{mn} \;=\; \pi^2 \sqrt{D / (\rho h)}\,
 *                       \left( m^2/a^2 + n^2/b^2 \right),
 *     \qquad m, n = 1, 2, 3, \ldots
 * @f]
 * with @f$ D = E h^3 / (12 (1 - \nu^2)) @f$.
 *
 * Mesh: @ref chladni::mesh::generate_flat_plate produces a hex-
 * triangulated chamfered-rectangle (a-by-b minus 2 corner triangles).
 * The 6 valence-3 corners of the chamfered shape are handled by L.5c.1's
 * augmentation; the rest is the regular Loop pipeline.
 *
 * BC enforcement: @em Hard Dirichlet on the z-component of every
 * boundary vertex (transverse displacement w = 0). The bending moment
 * is left free, matching Navier's BC on the plate. In-plane (x, y)
 * displacements are unconstrained — three rigid-body modes (x-trans,
 * y-trans, z-rot) survive and appear as zero eigenvalues, which we
 * skip when comparing to the analytic spectrum.
 *
 * Material / geometry: aluminum 0.5 m x 0.3 m x 1 mm — same fixture as
 * the @c [analytical][plate][simply_supported] test in
 * test_plate_simply_supported.cpp, so the analytic reference is
 * unambiguous.
 *
 * Tolerance: 5% relative on the lowest 4 modes at moderate
 * resolution. Observed at @c n_x = 20, @c n_y = 12 under the shipped
 * consistent-mass Loop assembler (commit 52579a9):
 *   mode (1, 1):  0.72% over analytic
 *   mode (2, 1):  1.68%
 *   mode (3, 1):  3.91%
 *   mode (1, 2):  4.49%
 * Error budget is mostly:
 *   (a) FEM discretisation on a 20x12 mesh (~21 cells per half-wave
 *       on the lowest mode, dropping to ~5 cells on (4, 1)),
 *   (b) the chamfered-rectangle shape vs a true rectangle (2 missing
 *       corner triangles ≈ 0.6% of plate area).
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

chladni::IsotropicMaterial aluminum()
{
    return {.youngs_modulus = 69.0e9,
            .poisson_ratio  = 0.33,
            .density        = 2700.0};
}

struct PlateModeResult {
    std::vector<double> rel_errs;        ///< length n_compare
    std::vector<double> fem_omegas;
    std::vector<double> analytic_omegas;
};

/// Run the SS rect plate FEM-vs-Leissa pipeline at the given mesh
/// resolution and return per-mode rel_errs (omega_fem vs omega_analytic).
/// Pins the structural properties along the way (3 rigid zero modes,
/// strictly positive omega_fem). Used by both the single-resolution
/// regression test and the convergence test.
PlateModeResult run_ss_plate_at(double a,
                                double b,
                                double h,
                                const chladni::IsotropicMaterial& mat,
                                int n_x,
                                int n_y,
                                std::size_t n_compare)
{
    const auto mesh = chladni::mesh::generate_flat_plate(a, b, n_x, n_y);
    const Eigen::Index n_v = mesh.V.rows();

    const auto sm   = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto K_full = chladni::shell::loop::assemble_stiffness_loop(
        mesh.V, mesh.F, sm);

    const auto vmasses = chladni::shell::lumped_vertex_masses(
        mesh.V, mesh.F, mat.density, h);
    const auto M_full = chladni::shell::assemble_mass_matrix(vmasses);

    const auto edges = chladni::shell::build_edges(mesh.F);
    std::vector<bool> is_bdry(static_cast<std::size_t>(n_v), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }
    std::vector<Eigen::Index> free_indices;
    free_indices.reserve(3 * n_v);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        for (int d = 0; d < 3; ++d) {
            const bool constrained =
                (d == 2) && is_bdry[static_cast<std::size_t>(v)];
            if (!constrained) free_indices.push_back(3 * v + d);
        }
    }
    const Eigen::Index n_free =
        static_cast<Eigen::Index>(free_indices.size());
    Eigen::SparseMatrix<double> P(3 * n_v, n_free);
    P.reserve(Eigen::VectorXi::Constant(n_free, 1));
    for (Eigen::Index k = 0; k < n_free; ++k) {
        P.insert(free_indices[static_cast<std::size_t>(k)], k) = 1.0;
    }
    Eigen::SparseMatrix<double> K_red = P.transpose() * K_full * P;
    Eigen::SparseMatrix<double> M_red = P.transpose() * M_full * P;

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        Eigen::MatrixXd(K_red), Eigen::MatrixXd(M_red),
        Eigen::ComputeEigenvectors | Eigen::Ax_lBx);
    REQUIRE(ges.info() == Eigen::Success);
    const Eigen::VectorXd evals = ges.eigenvalues();
    REQUIRE(evals.size() >= 3 + static_cast<Eigen::Index>(n_compare));

    // 3 rigid modes (x-trans, y-trans, z-rot) survive the SS BC.
    const double scale = std::abs(evals(evals.size() - 1));
    REQUIRE(std::abs(evals(0)) < 1e-6 * scale);
    REQUIRE(std::abs(evals(1)) < 1e-6 * scale);
    REQUIRE(std::abs(evals(2)) < 1e-6 * scale);

    PlateModeResult r;
    r.fem_omegas.reserve(n_compare);
    for (std::size_t k = 0; k < n_compare; ++k) {
        const Eigen::Index idx = 3 + static_cast<Eigen::Index>(k);
        REQUIRE(evals(idx) > 0.0);
        r.fem_omegas.push_back(std::sqrt(evals(idx)));
    }
    r.analytic_omegas =
        chladni::analytical::simply_supported_rect_plate_angular_frequencies(
            chladni::analytical::RectanglePlate{a, b, h}, mat, n_compare);
    r.rel_errs.reserve(n_compare);
    for (std::size_t k = 0; k < n_compare; ++k) {
        r.rel_errs.push_back(
            std::abs(r.fem_omegas[k] - r.analytic_omegas[k])
            / r.analytic_omegas[k]);
    }
    return r;
}

}  // namespace

TEST_CASE("Loop FEM vs Leissa: simply-supported rect plate (aluminum, 0.5m x 0.3m x 1mm)",
          "[shell][loop][modes][plate][simply_supported][analytic]")
{
    constexpr double a = 0.50;
    constexpr double b = 0.30;
    constexpr double h = 1.0e-3;
    constexpr int    n_x = 20;
    constexpr int    n_y = 12;
    constexpr std::size_t n_compare = 4;

    const auto r = run_ss_plate_at(a, b, h, aluminum(), n_x, n_y, n_compare);

    for (std::size_t k = 0; k < n_compare; ++k) {
        INFO("mode " << k
             << "  omega_fem = " << r.fem_omegas[k]
             << "  omega_analytic = " << r.analytic_omegas[k]
             << "  rel_err = " << r.rel_errs[k]);
        REQUIRE(r.rel_errs[k] < 0.05);
    }
}

TEST_CASE("Loop FEM vs Leissa: SS rect plate convergence under mesh refinement",
          "[shell][loop][modes][plate][simply_supported][convergence]")
{
    // Refine the mesh in two doublings of the cell count along each
    // axis, so the cell-edge length h shrinks by ~2x per step. For
    // O(h^2) plate-bending convergence we expect the rel_err to drop
    // by roughly 4x per doubling on the lowest mode (highest mode
    // resolution per wavelength).
    constexpr double a = 0.50;
    constexpr double b = 0.30;
    constexpr double h = 1.0e-3;
    constexpr std::size_t n_compare = 4;

    struct Resolution { int n_x; int n_y; };
    constexpr std::array<Resolution, 3> levels{{
        { 8,  5},
        {16, 10},
        {24, 15},
    }};

    std::array<double, 3> err_lowest{};
    for (std::size_t L = 0; L < levels.size(); ++L) {
        const auto r = run_ss_plate_at(a, b, h, aluminum(),
                                       levels[L].n_x, levels[L].n_y,
                                       n_compare);
        err_lowest[L] = r.rel_errs[0];

        // At every resolution the lowest 4 modes should at least be
        // physically plausible (positive, ascending, finite).
        for (std::size_t k = 0; k + 1 < n_compare; ++k) {
            CAPTURE(L, levels[L].n_x, levels[L].n_y, k);
            REQUIRE(r.fem_omegas[k] > 0.0);
            REQUIRE(r.fem_omegas[k + 1] >= r.fem_omegas[k]);
        }
    }

    // Strictly monotonic decrease in lowest-mode error as the mesh
    // refines.
    INFO("rel_err on (1, 1) at "
         << levels[0].n_x << "x" << levels[0].n_y << ": " << err_lowest[0]
         << " -> " << levels[1].n_x << "x" << levels[1].n_y << ": "
         << err_lowest[1]
         << " -> " << levels[2].n_x << "x" << levels[2].n_y << ": "
         << err_lowest[2]);
    REQUIRE(err_lowest[1] < err_lowest[0]);
    REQUIRE(err_lowest[2] < err_lowest[1]);

    // Coarse-to-fine ratio: at least 2.5x improvement between the
    // 8x5 and 24x15 meshes (a 9x cell-count refinement). For exact
    // O(h^2) we'd expect ~9x; we leave headroom for the chamfered-
    // rectangle shape error (which also scales as O(1/(n_x n_y)) but
    // with a different prefactor).
    REQUIRE(err_lowest[0] / err_lowest[2] > 2.5);
}
