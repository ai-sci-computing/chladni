/**
 * @file test_loop_cylinder_symmetry.cpp
 * @brief Audit whether the Loop K assembly respects the cylinder's
 *        discrete symmetries.
 *
 * The 2026-05-14 quadrature sweep surfaced an observation: on a
 * regularly triangulated cylinder, the Chladni mode shapes show skew
 * nodal lines where the continuous physics demands straight lines along
 * the generators. Two competing hypotheses:
 *
 *  (a) **K is direction-biased.** Some component of `accumulate_stiffness_at_point`
 *      treats v and w asymmetrically — a metric-tensor bug, a Voigt-
 *      convention swap, or a canonical-DOF-ordering effect that breaks
 *      cylinder symmetry.
 *
 *  (b) **The mesh itself is chiral.** `generate_cylinder` splits every
 *      quad with the same diagonal R_j[i] -> R_{j+1}[i+1], so the mesh
 *      inherits only the Z_n azimuthal subgroup of the continuous
 *      cylinder's O(2) × Z_2 symmetry. Axial reflection is broken
 *      *by construction*; the FEM is faithful to its mesh.
 *
 * This file separates the two:
 *
 *  Test 1: K should commute with the discrete azimuthal rotation
 *          (vertex permutation i -> (i+1) mod n_around combined with
 *           the corresponding 3x3 SO(3) rotation by 2π/n_around).
 *          If this holds, the FEM respects the symmetry the mesh
 *          *has*; the skew nodal lines come from hypothesis (b).
 *
 *  Test 2: K does NOT commute with the discrete axial reflection
 *          (z -> -z combined with the appropriate vertex permutation).
 *          This pins hypothesis (b) — the broken axial reflection is a
 *          measurable property of the mesh, not the assembler.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

namespace cs   = chladni::shell;
namespace cmsh = chladni::mesh;

namespace {

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

cs::ShellMaterial calibrated_steel()
{
    return cs::shell_material_from_isotropic(steel(), /*thickness=*/1.0e-3);
}

// Build the (3N x 3N) operator that azimuthally rotates the cylinder
// by 2π/n_around: vertex (i, j) at column-index j*n_around+i goes to
// the column where vertex ((i+1) mod n_around, j) sits, and each
// 3-vector of components is rotated by 2π/n_around around the cylinder
// axis (z).
Eigen::SparseMatrix<double> azimuthal_rotation_operator(
    Eigen::Index n_around,
    Eigen::Index n_rings)
{
    const Eigen::Index n_v   = n_around * n_rings;
    const Eigen::Index dim   = 3 * n_v;
    const double      theta  = 2.0 * std::numbers::pi_v<double> / static_cast<double>(n_around);
    const double      cosT   = std::cos(theta);
    const double      sinT   = std::sin(theta);

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(9 * n_v);
    for (Eigen::Index j = 0; j < n_rings; ++j) {
        for (Eigen::Index i = 0; i < n_around; ++i) {
            const Eigen::Index src = j * n_around + i;
            const Eigen::Index dst = j * n_around + ((i + 1) % n_around);
            // 3x3 z-axis rotation block at (dst, src).
            trips.emplace_back(3 * dst + 0, 3 * src + 0,  cosT);
            trips.emplace_back(3 * dst + 0, 3 * src + 1, -sinT);
            trips.emplace_back(3 * dst + 1, 3 * src + 0,  sinT);
            trips.emplace_back(3 * dst + 1, 3 * src + 1,  cosT);
            trips.emplace_back(3 * dst + 2, 3 * src + 2,  1.0);
        }
    }
    Eigen::SparseMatrix<double> T(dim, dim);
    T.setFromTriplets(trips.begin(), trips.end());
    T.makeCompressed();
    return T;
}

// Build the (3N x 3N) operator that axially reflects the cylinder:
// vertex (i, j) -> (i, n_rings - 1 - j), z component negated.
Eigen::SparseMatrix<double> axial_reflection_operator(
    Eigen::Index n_around,
    Eigen::Index n_rings)
{
    const Eigen::Index n_v   = n_around * n_rings;
    const Eigen::Index dim   = 3 * n_v;

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(3 * n_v);
    for (Eigen::Index j = 0; j < n_rings; ++j) {
        const Eigen::Index j_flip = n_rings - 1 - j;
        for (Eigen::Index i = 0; i < n_around; ++i) {
            const Eigen::Index src = j      * n_around + i;
            const Eigen::Index dst = j_flip * n_around + i;
            trips.emplace_back(3 * dst + 0, 3 * src + 0,  1.0);
            trips.emplace_back(3 * dst + 1, 3 * src + 1,  1.0);
            trips.emplace_back(3 * dst + 2, 3 * src + 2, -1.0);
        }
    }
    Eigen::SparseMatrix<double> T(dim, dim);
    T.setFromTriplets(trips.begin(), trips.end());
    T.makeCompressed();
    return T;
}

double max_abs(const Eigen::SparseMatrix<double>& M)
{
    return Eigen::MatrixXd(M).cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("cylinder M respects discrete azimuthal rotation",
          "[shell][loop][symmetry][cylinder][audit]")
{
    // If [M, T_azim] is not machine epsilon, the generalized eigenproblem
    // K v = ω² M v breaks Z_n symmetry through M even though K is
    // perfectly Z_n-invariant. This would explain the cylinder
    // doublet pairs being split by 1-2% (not numerical noise) — and
    // would point at M assembly as the bug, not K.
    constexpr int  n_around = 12;
    constexpr int  n_along  = 4;
    constexpr double R = 0.10;
    constexpr double L = 0.20;

    const auto mesh = cmsh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_rings = static_cast<Eigen::Index>(n_along) + 1;

    cs::LoopAssembler asm_default;
    const auto M = asm_default.assemble_M(
        mesh.V, mesh.F, /*surface_density=*/7.85);

    const auto T = azimuthal_rotation_operator(n_around, n_rings);

    Eigen::SparseMatrix<double> commutator = M * T - T * M;
    const double comm_norm = max_abs(commutator);
    const double m_scale   = max_abs(M);

    INFO("n_around=" << n_around << "  n_along=" << n_along
         << "  M scale = " << m_scale
         << "  ||[M, T_azim]||_inf = " << comm_norm
         << "  ratio = " << (comm_norm / m_scale));
    std::cout << "[cylinder symmetry] M azimuthal: M scale = " << m_scale
              << "  commutator = " << comm_norm
              << "  ratio = " << (comm_norm / m_scale) << '\n';

    REQUIRE(comm_norm <= 1.0e-9 * m_scale);
}

TEST_CASE("cylinder K respects discrete azimuthal rotation",
          "[shell][loop][symmetry][cylinder][audit]")
{
    constexpr int  n_around = 12;
    constexpr int  n_along  = 4;
    constexpr double R = 0.10;
    constexpr double L = 0.20;

    const auto mesh = cmsh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_rings = static_cast<Eigen::Index>(n_along) + 1;

    cs::LoopAssembler asm_default;
    const auto K = asm_default.assemble_K(mesh.V, mesh.F, calibrated_steel());

    const auto T = azimuthal_rotation_operator(n_around, n_rings);

    // K T == T K means K commutes with the symmetry, which is the
    // FEM-respects-mesh-symmetry contract.
    Eigen::SparseMatrix<double> commutator = K * T - T * K;
    const double comm_norm = max_abs(commutator);
    const double k_scale   = max_abs(K);

    INFO("n_around=" << n_around << "  n_along=" << n_along
         << "  K scale = " << k_scale
         << "  ||[K, T_azim]||_inf = " << comm_norm
         << "  ratio = " << (comm_norm / k_scale));
    std::cout << "[cylinder symmetry] azimuthal: K scale = " << k_scale
              << "  commutator = " << comm_norm
              << "  ratio = " << (comm_norm / k_scale) << '\n';

    // Tight bound: K must commute with the azimuthal rotation to
    // within numerical round-off (Eigen's sparse arithmetic on order
    // 3*n_v ~ 200 dof gives noise around 1e-12 * scale).
    REQUIRE(comm_norm <= 1.0e-9 * k_scale);
}

TEST_CASE("cylinder K does NOT respect axial reflection — mesh is chiral",
          "[shell][loop][symmetry][cylinder][audit]")
{
    // The "/" diagonal in every quad of generate_cylinder breaks axial
    // reflection. Reflecting z -> -z without re-triangulating sends
    // the "/" diagonal to "\", a different mesh. So K should NOT
    // commute with axial reflection — and the failure ratio should be
    // O(1), not numerical noise. If it WERE small, the mesh would
    // accidentally be achiral and our explanation of skew nodal lines
    // would collapse.
    constexpr int  n_around = 12;
    constexpr int  n_along  = 4;
    constexpr double R = 0.10;
    constexpr double L = 0.20;

    const auto mesh = cmsh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_rings = static_cast<Eigen::Index>(n_along) + 1;

    cs::LoopAssembler asm_default;
    const auto K = asm_default.assemble_K(mesh.V, mesh.F, calibrated_steel());

    const auto R_ax = axial_reflection_operator(n_around, n_rings);

    Eigen::SparseMatrix<double> commutator = K * R_ax - R_ax * K;
    const double comm_norm = max_abs(commutator);
    const double k_scale   = max_abs(K);

    INFO("n_around=" << n_around << "  n_along=" << n_along
         << "  K scale = " << k_scale
         << "  ||[K, R_axial]||_inf = " << comm_norm
         << "  ratio = " << (comm_norm / k_scale));
    std::cout << "[cylinder symmetry] axial: K scale = " << k_scale
              << "  commutator = " << comm_norm
              << "  ratio = " << (comm_norm / k_scale) << '\n';

    // Pin: the commutator should be O(1) relative to K, NOT noise.
    REQUIRE(comm_norm > 1.0e-4 * k_scale);
}
