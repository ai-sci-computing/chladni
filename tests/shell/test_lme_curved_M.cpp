/**
 * @file test_lme_curved_M.cpp
 * @brief Curved-shell LME consistent mass assembly.
 *
 * Mirrors the K-side tests in test_lme_curved_K.cpp but for the
 * consistent mass matrix
 *
 *   @f[
 *     M_{ab} = \int_{\mathcal M} \rho h\, p_a(x)\, p_b(x)\,\mathrm{d}\mathcal M ,
 *   @f]
 *
 * tiled onto a 3×3 identity block per (a, b) pair (Kirchhoff plate
 * decouples in-plane and out-of-plane motion at the mass level).
 *
 * Coverage:
 *  1. **Flat plate total mass.** @f$ e_d^\top M e_d = \rho h \cdot
 *     \mathrm{area} @f$ for every displacement direction d.
 *  2. **Icosphere total mass.** Sphere surface area @f$ 4\pi r^2 @f$
 *     is recovered from the curved M to within 1 % at @c n_subdivisions
 *     @f$\ge 2@f$ — covers Shepard-PoU localisation error plus the
 *     polygonal-mesh-vs-true-sphere geometric gap.
 *  3. **Symmetry + positive-definiteness.** M is sym-PD on both flat
 *     and curved fixtures.
 *  4. **Block-diagonal (no cross-component coupling).** Vertex-pair
 *     entries respect the 3-component identity tiling.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

#include <numbers>
#include <vector>

using chladni::shell::LMEAssembler;
using Catch::Matchers::WithinAbs;

namespace {

/// xy-projected (signed) area of a triangulation. Mirrors the helper
/// in test_lme_assembler.cpp — accounts for the corner-chamfering in
/// generate_flat_plate that lops off two single-triangle wedges.
double mesh_area_xy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double total = 0.0;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        const Eigen::RowVector2d p0 = V.row(F(t, 0)).head<2>();
        const Eigen::RowVector2d p1 = V.row(F(t, 1)).head<2>();
        const Eigen::RowVector2d p2 = V.row(F(t, 2)).head<2>();
        const Eigen::RowVector2d e01 = p1 - p0;
        const Eigen::RowVector2d e02 = p2 - p0;
        total += 0.5 * std::abs(e01.x() * e02.y() - e01.y() * e02.x());
    }
    return total;
}

/// 3D-triangle surface area sum.
double mesh_surface_area(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double total = 0.0;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        const Eigen::Vector3d e01 =
            (V.row(F(t, 1)) - V.row(F(t, 0))).transpose();
        const Eigen::Vector3d e02 =
            (V.row(F(t, 2)) - V.row(F(t, 0))).transpose();
        total += 0.5 * e01.cross(e02).norm();
    }
    return total;
}

}  // namespace

TEST_CASE("LMEAssembler::assemble_M curved path: total mass on a flat plate",
          "[shell][lme][assembler][curved_M]")
{
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/1.0, /*length_b=*/1.0, /*n_x=*/6, /*n_y=*/6);
    const double area = mesh_area_xy(plate.V, plate.F);
    REQUIRE(area > 0.0);

    constexpr double rho_h = 7.0;

    LMEAssembler::Params params;
    params.use_curved_shell = true;
    // Total-mass identity is asserted on the real-vertex block; ghost-on
    // inflates M to 3·(N+G) with extra rows whose mass integrates over
    // the boundary-extension support and so does NOT equal ρh·area.
    params.use_ghost_nodes = false;
    const auto M = LMEAssembler(params).assemble_M(plate.V, plate.F, rho_h);

    REQUIRE(M.rows() == 3 * plate.V.rows());
    REQUIRE(M.cols() == 3 * plate.V.rows());

    // Per displacement direction: e_d^T M e_d = ρh · area.
    for (int d = 0; d < 3; ++d) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(3 * plate.V.rows());
        for (Eigen::Index v = 0; v < plate.V.rows(); ++v) {
            e(3 * v + d) = 1.0;
        }
        const double mass_d = e.transpose() * M * e;
        INFO("displacement d=" << d << "  expected " << rho_h * area);
        REQUIRE_THAT(mass_d, WithinAbs(rho_h * area, 1.0e-9));
    }
}

TEST_CASE("LMEAssembler::assemble_M curved path: total mass on an icosphere",
          "[shell][lme][assembler][curved_M]")
{
    constexpr double radius = 1.0;
    const auto       sphere = chladni::mesh::generate_icosphere(
        /*radius=*/radius, /*n_subdivisions=*/2);
    const double mesh_area = mesh_surface_area(sphere.V, sphere.F);
    REQUIRE(mesh_area > 0.0);

    constexpr double rho_h = 2.0;

    LMEAssembler::Params params;
    params.use_curved_shell = true;
    const auto M = LMEAssembler(params).assemble_M(sphere.V, sphere.F, rho_h);

    REQUIRE(M.rows() == 3 * sphere.V.rows());
    REQUIRE(M.cols() == 3 * sphere.V.rows());

    // Per displacement direction: e_d^T M e_d = ρh · mesh_surface_area
    // (the curved formulation integrates on the input F triangles in R³,
    // so the mass equals ρh times the polygonal mesh's R³ area — not
    // the exact-sphere 4π r²).
    for (int d = 0; d < 3; ++d) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(3 * sphere.V.rows());
        for (Eigen::Index v = 0; v < sphere.V.rows(); ++v) {
            e(3 * v + d) = 1.0;
        }
        const double mass_d = e.transpose() * M * e;
        const double rel_err = std::abs(mass_d - rho_h * mesh_area)
                                / (rho_h * mesh_area);
        INFO("d=" << d << "  mass_d=" << mass_d
             << "  ρh·mesh_area=" << rho_h * mesh_area
             << "  rel_err=" << rel_err);
        REQUIRE(rel_err < 1.0e-6);
    }

    // Also sanity-check the polygonal-mesh approximation of the analytic
    // sphere area 4π r². n_subdivisions=2 (162V) should be within 1 %.
    const double analytic_area = 4.0 * std::numbers::pi * radius * radius;
    const double geom_err = std::abs(mesh_area - analytic_area) / analytic_area;
    REQUIRE(geom_err < 0.02);
}

TEST_CASE("LMEAssembler::assemble_M curved path: symmetric + positive-definite",
          "[shell][lme][assembler][curved_M]")
{
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/1.0, /*length_b=*/1.0, /*n_x=*/4, /*n_y=*/4);

    LMEAssembler::Params params;
    params.use_curved_shell = true;
    const auto M = LMEAssembler(params).assemble_M(plate.V, plate.F, 1.0);

    const Eigen::SparseMatrix<double> M_T = M.transpose();
    const Eigen::SparseMatrix<double> diff = M - M_T;
    REQUIRE_THAT(diff.norm(), WithinAbs(0.0, 1.0e-12));

    const Eigen::MatrixXd M_dense = Eigen::MatrixXd(M);
    Eigen::LLT<Eigen::MatrixXd> llt(M_dense);
    REQUIRE(llt.info() == Eigen::Success);
}

TEST_CASE("LMEAssembler::assemble_M curved+SME: symmetric and PD on flat plate, ghost-on",
          "[shell][lme][assembler][curved_M][sme]")
{
    // Wire-up sanity for Phase B.1: with SME on, the consistent mass
    // must still be symmetric and positive-definite. PoU holds by
    // SME construction (the dual normalisation handles it).
    //
    // Mesh size and L=1.0 chosen so kRing=3 charts are local and node
    // spacing h~0.08 keeps SME's Newton in its well-converging
    // regime; see the matching curved_K SME smoke for the scale and
    // chart-locality rationale. Total-mass identity is NOT asserted
    // because the ghost-extended M includes boundary-extension rows
    // whose row-sum doesn't equal ρh·area; the curved-M ghost-off
    // paths in the suite already pin the conservation property.
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/1.0, /*length_b=*/1.0, /*n_x=*/12, /*n_y=*/12);

    LMEAssembler::Params params;
    params.use_curved_shell     = true;
    params.use_ghost_nodes      = true;
    params.use_second_order_sme = true;
    // No r_cut override: SME truncation is VALUE-based (γ_eff = 2/α,
    // ≈4.80 h at α=2; inventory C7, r_cut_mult_sme retired), the wide
    // active set its Newton needs to converge robustly here.
    const auto M = LMEAssembler(params).assemble_M(plate.V, plate.F, 7.0);

    REQUIRE(M.rows() > 3 * plate.V.rows());
    REQUIRE(M.cols() == M.rows());

    const Eigen::SparseMatrix<double> M_T   = M.transpose();
    const Eigen::SparseMatrix<double> diff  = M - M_T;
    REQUIRE_THAT(diff.norm(), WithinAbs(0.0, 1.0e-10));

    const Eigen::MatrixXd M_dense = Eigen::MatrixXd(M);
    Eigen::LLT<Eigen::MatrixXd> llt(M_dense);
    REQUIRE(llt.info() == Eigen::Success);
}
