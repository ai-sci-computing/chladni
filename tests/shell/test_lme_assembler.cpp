/**
 * @file test_lme_assembler.cpp
 * @brief Unit tests for @ref chladni::shell::LMEAssembler.
 *
 * Coverage:
 *
 *  - **Mass conservation** — the consistent LME mass matrix integrates
 *    the constant @f$ \rho h \cdot 1 @f$ to exactly the surface area on
 *    a simple unit-square fixture. With wide enough @c r_cut the
 *    partition-of-unity holds at every quadrature point and the
 *    integral is exact up to round-off (the 7-point Dunavant rule is
 *    exact on the integrand @f$ \sum_{a,b} p_a p_b = 1 @f$).
 *  - **Symmetry** — assembled @f$ M @f$ is byte-symmetric.
 *  - **Positive-definiteness** — dense Cholesky succeeds.
 *  - **Block-diagonal structure** — no coupling between displacement
 *    components in the 3·n_V layout. Kirchhoff plate decouples
 *    in-plane and out-of-plane motion at the mass level (off-diagonal
 *    blocks identically zero) and the three diagonal blocks equal each
 *    other.
 *  - **Curved-shell rejection** — passing an icosphere mesh throws
 *    @c std::invalid_argument until the wPCA / Shepard PU extension
 *    lands.
 *  - **assemble_K stub still throws** — step 5 of the implementation
 *    plan has not landed yet; calling @c assemble_K must surface a
 *    @c std::runtime_error so callers fail loudly.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <cmath>
#include <limits>
#include <stdexcept>

using chladni::shell::LMEAssembler;
using chladni::shell::ShellMaterial;
using Catch::Matchers::WithinAbs;

namespace {

/// Stack the canonical x/y/z component vectors of the 3·n_V layout:
/// @c e[d](3·a + d) = 1 for @c a in @c [0, n_v), zero elsewhere.
std::array<Eigen::VectorXd, 3> component_axis_vectors(Eigen::Index n_v)
{
    std::array<Eigen::VectorXd, 3> e;
    for (int d = 0; d < 3; ++d) {
        e[static_cast<std::size_t>(d)] =
            Eigen::VectorXd::Zero(3 * n_v);
        for (Eigen::Index a = 0; a < n_v; ++a) {
            e[static_cast<std::size_t>(d)](3 * a + d) = 1.0;
        }
    }
    return e;
}

/// Sum of the (xy-projected) signed triangle areas — the integrated
/// surface area of the mesh treated as a flat plate. Independent of
/// what we put in @c V.col(2); LMEAssembler dispatches to 2D anyway.
double mesh_area_xy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double a = 0.0;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        const Eigen::Vector2d v0 = V.row(F(t, 0)).head<2>().transpose();
        const Eigen::Vector2d v1 = V.row(F(t, 1)).head<2>().transpose();
        const Eigen::Vector2d v2 = V.row(F(t, 2)).head<2>().transpose();
        const Eigen::Vector2d e1 = v1 - v0;
        const Eigen::Vector2d e2 = v2 - v0;
        a += 0.5 * std::abs(e1.x() * e2.y() - e1.y() * e2.x());
    }
    return a;
}

}  // namespace

TEST_CASE("LMEAssembler::assemble_M: total mass on a unit square",
          "[lme][assembler][mass]")
{
    // 4x4 grid → 16 vertices, 18 triangles. Small enough for fast
    // tests; large enough that every node has neighbours in every
    // direction.
    // Note: generate_flat_plate omits two opposite-corner vertices
    // (n_x, 0) and (0, n_y) along with the one triangle each was a
    // corner of (mesh.cpp:269-291). The remaining plate area is
    // therefore length_a · length_b − 2 · (one-triangle area). Read
    // the integrated area off F directly so the assertion stays
    // numerically exact.
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/1.0, /*length_b=*/1.0, /*n_x=*/4, /*n_y=*/4);
    const double area = mesh_area_xy(plate.V, plate.F);

    constexpr double rho_h = 1.0;  // surface density (kg/m²)
    // Flat path here: this test asserts total-mass = ρh·area on the
    // real-vertex block. The default-curved+ghost path inflates M to
    // 3·(N+G) and adds ghost-block mass that doesn't equal ρh·area;
    // total-mass conservation on the ghost-on path is gated by the
    // dedicated curved-M tests in test_lme_curved_M.cpp.
    LMEAssembler::Params params;
    params.use_curved_shell = false;
    LMEAssembler asm0(params);

    const auto M = asm0.assemble_M(plate.V, plate.F, rho_h);
    REQUIRE(M.rows() == 3 * plate.V.rows());
    REQUIRE(M.cols() == 3 * plate.V.rows());

    // Per displacement direction: total mass = e_d^T M e_d, with e_d
    // the unit-z vector. Independent of d by block-diagonal structure.
    const auto e = component_axis_vectors(plate.V.rows());
    for (int d = 0; d < 3; ++d) {
        const double mass_d = e[static_cast<std::size_t>(d)].transpose()
                              * M * e[static_cast<std::size_t>(d)];
        INFO("displacement component d = " << d
             << "; expected ρh·area = " << rho_h * area);
        REQUIRE_THAT(mass_d, WithinAbs(rho_h * area, 1e-10));
    }
}

TEST_CASE("LMEAssembler::assemble_M: symmetric, positive-definite",
          "[lme][assembler][mass]")
{
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/1.0, /*length_b=*/1.0, /*n_x=*/4, /*n_y=*/4);

    LMEAssembler asm0;
    const auto M = asm0.assemble_M(plate.V, plate.F, /*surface_density=*/1.0);

    // ----- symmetry -------------------------------------------------
    const Eigen::SparseMatrix<double> diff = M - Eigen::SparseMatrix<double>(M.transpose());
    REQUIRE_THAT(diff.norm(), WithinAbs(0.0, 1e-12));

    // ----- positive-definiteness via dense Cholesky -----------------
    // Small mesh — dense LLT is fine and gives a clean success/fail
    // signal without parameter tuning of an iterative solver.
    const Eigen::MatrixXd M_dense = Eigen::MatrixXd(M);
    Eigen::LLT<Eigen::MatrixXd> llt(M_dense);
    REQUIRE(llt.info() == Eigen::Success);
}

TEST_CASE("LMEAssembler::assemble_M: block-diagonal structure",
          "[lme][assembler][mass]")
{
    // The three displacement components share a single scalar mass
    // matrix and don't couple: M[3a+d_i, 3b+d_j] = 0 if d_i != d_j,
    // and M[3a+0, 3b+0] = M[3a+1, 3b+1] = M[3a+2, 3b+2].
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 4, 4);
    LMEAssembler asm0;
    const auto M = asm0.assemble_M(plate.V, plate.F, /*surface_density=*/1.0);

    // The default assembler is curved + ghost-on, so M is sized to the
    // GHOST-EXTENDED dof count (3*n_ext), not 3*plate.V.rows(). Size the
    // probe vectors to M itself — using plate.V.rows() under-sizes them
    // and the e^T M products read past the end of e (Eigen catches the
    // size mismatch in debug; release silently reads OOB heap garbage).
    const Eigen::Index n_dof = M.rows() / 3;
    const auto e = component_axis_vectors(n_dof);

    // Off-diagonal e_d^T M e_d': zero (no cross-component coupling).
    for (int di = 0; di < 3; ++di) {
        for (int dj = di + 1; dj < 3; ++dj) {
            const double coupling = e[static_cast<std::size_t>(di)].transpose()
                                    * M * e[static_cast<std::size_t>(dj)];
            INFO("cross-coupling (" << di << "," << dj << ")");
            REQUIRE_THAT(coupling, WithinAbs(0.0, 1e-12));
        }
    }

    // Per-entry block identity: M[3a, 3b] = M[3a+1, 3b+1] = M[3a+2, 3b+2].
    const Eigen::MatrixXd M_dense = Eigen::MatrixXd(M);
    for (Eigen::Index a = 0; a < n_dof; ++a) {
        for (Eigen::Index b = 0; b < n_dof; ++b) {
            const double m_xx = M_dense(3 * a + 0, 3 * b + 0);
            const double m_yy = M_dense(3 * a + 1, 3 * b + 1);
            const double m_zz = M_dense(3 * a + 2, 3 * b + 2);
            REQUIRE_THAT(m_xx - m_yy, WithinAbs(0.0, 1e-12));
            REQUIRE_THAT(m_xx - m_zz, WithinAbs(0.0, 1e-12));
        }
    }
}

TEST_CASE("LMEAssembler::assemble_M: rejects curved-shell input on the flat path",
          "[lme][assembler][mass]")
{
    // The legacy flat-plate path (use_curved_shell=false) requires
    // strictly planar input and throws on a curved mesh. The
    // default-curved path handles both.
    const auto sphere = chladni::mesh::generate_icosphere(
        /*radius=*/1.0, /*n_subdivisions=*/2);
    LMEAssembler::Params params;
    params.use_curved_shell = false;
    LMEAssembler asm_flat(params);
    REQUIRE_THROWS_AS(
        asm_flat.assemble_M(sphere.V, sphere.F, /*surface_density=*/1.0),
        std::invalid_argument);
}

TEST_CASE("LMEAssembler::assemble_M: input validation",
          "[lme][assembler][mass]")
{
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 3, 3);
    LMEAssembler asm0;

    SECTION("nonpositive surface density throws")
    {
        REQUIRE_THROWS_AS(
            asm0.assemble_M(plate.V, plate.F, 0.0),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            asm0.assemble_M(plate.V, plate.F, -1.0),
            std::invalid_argument);
    }

    SECTION("empty mesh throws")
    {
        Eigen::MatrixXd Vempty(0, 3);
        Eigen::MatrixXi Fempty(0, 3);
        REQUIRE_THROWS_AS(
            asm0.assemble_M(Vempty, Fempty, 1.0),
            std::invalid_argument);
    }
}

TEST_CASE("LMEAssembler::assemble_K: symmetric and rigid-body kernel",
          "[lme][assembler][stiffness]")
{
    // Three rigid-body modes of a Kirchhoff plate: constant z-disp,
    // linear x (a tilt about the y-axis), linear y (about the x-axis).
    // Each must lie in ker(K) — the integrand
    //   D · [Δu·Δv − (1−ν)(...)]
    // vanishes whenever u has identically zero Hessian. PoU and affine
    // reproduction of the LME basis lift this to the coefficient layer:
    //   constant z  → α[3a+2] = 1       (Σ_a ∇²p_a = 0)
    //   linear x    → α[3a+2] = x_a     (Σ_a x_a ∇²p_a = 0)
    //   linear y    → α[3a+2] = y_a     (Σ_a y_a ∇²p_a = 0)
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 4, 4);
    const Eigen::Index n_v = plate.V.rows();

    // Tight Newton tolerance so the rigid-body kernel residual is
    // dominated by round-off rather than Newton drift propagating
    // through J^{-1} into each Hessian. With ||J^{-1}|| ~ 50 near the
    // boundary and ~50 quadrature points contributing, the per-mode
    // residual scales like ||J^{-1}|| · newton_tol · n_quad ≈
    // 50 · 1e-13 · 50 ≈ 2.5e-10.
    LMEAssembler::Params p;
    // Flat path — test asserts K shape = 3·n_v. Ghost-on inflates.
    p.use_curved_shell = false;
    p.newton_tol = 1e-13;
    p.newton_max_iters = 60;
    LMEAssembler asm0(p);
    const ShellMaterial mat{
        /*k_L=*/0.0,            // membrane unused — not assembled yet
        /*k_B=*/1.0,
        /*poisson_ratio=*/0.3};
    const auto K = asm0.assemble_K(plate.V, plate.F, mat);
    REQUIRE(K.rows() == 3 * n_v);
    REQUIRE(K.cols() == 3 * n_v);

    // ----- symmetry -------------------------------------------------
    const Eigen::SparseMatrix<double> diff =
        K - Eigen::SparseMatrix<double>(K.transpose());
    REQUIRE_THAT(diff.norm(), WithinAbs(0.0, 1e-12));

    // ----- rigid-body modes lie in kernel ---------------------------
    // Build coefficient vectors for the three modes on the z-component.
    auto z_layout = [n_v](auto f) {
        Eigen::VectorXd a = Eigen::VectorXd::Zero(3 * n_v);
        for (Eigen::Index k = 0; k < n_v; ++k) a(3 * k + 2) = f(k);
        return a;
    };
    const Eigen::VectorXd alpha_one = z_layout(
        [](Eigen::Index) { return 1.0; });
    const Eigen::VectorXd alpha_x   = z_layout(
        [&plate](Eigen::Index k) { return plate.V(k, 0); });
    const Eigen::VectorXd alpha_y   = z_layout(
        [&plate](Eigen::Index k) { return plate.V(k, 1); });

    // K · α should be zero up to the Newton-noise floor — see the
    // scaling estimate above. With Newton tol 1e-13, the residual
    // observed is ~1e-10; check a slightly looser bound to leave
    // headroom for the per-mesh constants.
    INFO("constant z-disp coefficient vector");
    REQUIRE_THAT((K * alpha_one).norm(), WithinAbs(0.0, 1e-8));
    INFO("linear z = x coefficient vector");
    REQUIRE_THAT((K * alpha_x).norm(),   WithinAbs(0.0, 1e-8));
    INFO("linear z = y coefficient vector");
    REQUIRE_THAT((K * alpha_y).norm(),   WithinAbs(0.0, 1e-8));
}

TEST_CASE("LMEAssembler::assemble_K: 6-dim rigid-body kernel with membrane on",
          "[lme][assembler][stiffness]")
{
    // With both membrane (k_L > 0) and bending (k_B > 0) active on a
    // flat plate, LMEAssembler::assemble_K must reproduce the standard
    // shell-FEM rigid-body kernel: exactly six zero modes spanning the
    // rigid-translation and rigid-rotation subspace of @f$\mathbb R^3@f$.
    //
    // At @c z = 0 the six rigid-body fields collapse to
    //
    //   T_x:  u = (1, 0, 0)       → α[3a + 0] = 1
    //   T_y:  u = (0, 1, 0)       → α[3a + 1] = 1
    //   T_z:  u = (0, 0, 1)       → α[3a + 2] = 1
    //   R_x (about x-axis): u = (0, 0,  y)  → α[3a + 2] =  y_a
    //   R_y (about y-axis): u = (0, 0, -x)  → α[3a + 2] = -x_a
    //   R_z (about z-axis): u = (-y, x, 0)  → α[3a + 0] = -y_a,
    //                                         α[3a + 1] =  x_a
    //
    // T_z, R_x, R_y vanish identically against the **bending** integrand
    // (constant + affine in z give zero Hessian). T_x, T_y vanish against
    // the **membrane** integrand (constant in-plane → zero strain).
    // R_z is the well-known rigid in-plane rotation: ∂u_x/∂x = 0,
    // ∂u_y/∂y = 0, and ∂u_x/∂y + ∂u_y/∂x = (-1) + 1 = 0 → zero strain.
    //
    // The coefficient-space representation matches the displacement
    // because LME has affine reproduction (Σ_a p_a x_a = x): a linear
    // function f(x) is interpolated exactly by setting α_a = f(x_a).
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 4, 4);
    const Eigen::Index n_v = plate.V.rows();

    LMEAssembler::Params p;
    // Flat path — test asserts K shape = 3·n_v. Ghost-on inflates.
    p.use_curved_shell = false;
    p.newton_tol       = 1e-13;
    p.newton_max_iters = 60;
    LMEAssembler asm0(p);
    const ShellMaterial mat{
        /*k_L=*/1.0,
        /*k_B=*/1.0,
        /*poisson_ratio=*/0.3};
    const auto K = asm0.assemble_K(plate.V, plate.F, mat);
    REQUIRE(K.rows() == 3 * n_v);
    REQUIRE(K.cols() == 3 * n_v);

    Eigen::MatrixXd V_rigid = Eigen::MatrixXd::Zero(3 * n_v, 6);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        const double x = plate.V(a, 0);
        const double y = plate.V(a, 1);
        // T_x, T_y, T_z
        V_rigid(3 * a + 0, 0) = 1.0;
        V_rigid(3 * a + 1, 1) = 1.0;
        V_rigid(3 * a + 2, 2) = 1.0;
        // R_x: u_z = y
        V_rigid(3 * a + 2, 3) =  y;
        // R_y: u_z = -x
        V_rigid(3 * a + 2, 4) = -x;
        // R_z: u_x = -y, u_y = x
        V_rigid(3 * a + 0, 5) = -y;
        V_rigid(3 * a + 1, 5) =  x;
    }

    const Eigen::MatrixXd KV = K * V_rigid;
    const char* labels[6] = {"T_x", "T_y", "T_z", "R_x", "R_y", "R_z"};
    for (int k = 0; k < 6; ++k) {
        INFO("rigid mode " << labels[k]);
        REQUIRE_THAT(KV.col(k).norm(), WithinAbs(0.0, 1e-7));
    }

    // Beyond the rigid subspace, no other mode may lie in ker(K) — a
    // proper shell K has rank @f$3 n_V - 6@f$ exactly. The "ghost"
    // in-plane stretches that fall through the legacy 3-tile-of-bending
    // construction must be **excluded** by membrane stiffness:
    //
    //   u_x = x   (uniform x-stretch)       → α[3a + 0] = x_a   (NOT rigid)
    //   u_y = y   (uniform y-stretch)       → α[3a + 1] = y_a   (NOT rigid)
    //   u_x = y, u_y = x  (symmetric shear) → α[3a + 0] = y_a,
    //                                         α[3a + 1] = x_a   (NOT rigid)
    //
    // The membrane integrand `B^T D B` is strictly positive on each of
    // these (e.g. u_x = x gives ε_xx = 1; D · [1, 0, 0]^T is nonzero),
    // so each of these alpha vectors must produce a nonzero strain
    // energy α^T K α > 0.
    Eigen::MatrixXd V_spurious = Eigen::MatrixXd::Zero(3 * n_v, 3);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        const double x = plate.V(a, 0);
        const double y = plate.V(a, 1);
        V_spurious(3 * a + 0, 0) = x;            // u_x = x
        V_spurious(3 * a + 1, 1) = y;            // u_y = y
        V_spurious(3 * a + 0, 2) = y;            // u_x = y
        V_spurious(3 * a + 1, 2) = x;            //   + u_y = x
    }
    const char* spurious_labels[3] = {"u_x=x", "u_y=y", "symmetric shear"};
    for (int k = 0; k < 3; ++k) {
        const Eigen::VectorXd alpha = V_spurious.col(k);
        const double energy = alpha.transpose() * K * alpha;
        INFO("spurious in-plane mode " << spurious_labels[k]
             << " strain energy = " << energy);
        REQUIRE(energy > 1e-4);
    }
}

TEST_CASE("LMEAssembler::assemble_K: positive-semi-definite",
          "[lme][assembler][stiffness]")
{
    // Dense self-adjoint eigendecomposition on a coarse mesh — small
    // enough to be fast (3·n_V = 42 for the 4×4 plate). The smallest
    // eigenvalue should be ≥ -ε, where the ε margin tracks the noise
    // floor of the rigid-body modes (in coefficient space, those
    // expected zero eigenvalues drift by Newton-tol-times-conditioning).
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 4, 4);
    LMEAssembler asm0;
    const ShellMaterial mat{
        /*k_L=*/0.0, /*k_B=*/1.0, /*poisson_ratio=*/0.3};
    const auto K = asm0.assemble_K(plate.V, plate.F, mat);

    const Eigen::MatrixXd Kd = Eigen::MatrixXd(K);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd);
    REQUIRE(es.info() == Eigen::Success);

    const double lam_min = es.eigenvalues().minCoeff();
    INFO("smallest eigenvalue = " << lam_min);
    REQUIRE(lam_min > -1e-10);
}

TEST_CASE("LMEAssembler::assemble_K: input validation",
          "[lme][assembler][stiffness]")
{
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 3, 3);
    LMEAssembler asm0;

    SECTION("nonpositive bending stiffness throws")
    {
        const ShellMaterial bad{0.0, 0.0, 0.3};
        REQUIRE_THROWS_AS(
            asm0.assemble_K(plate.V, plate.F, bad),
            std::invalid_argument);
    }

    SECTION("curved-shell input throws on the flat path")
    {
        // Flat path (use_curved_shell=false) rejects curved input.
        // The default-curved path accepts it; this section gates
        // the legacy path only.
        const auto sphere = chladni::mesh::generate_icosphere(1.0, 2);
        const ShellMaterial mat{0.0, 1.0, 0.3};
        LMEAssembler::Params params;
        params.use_curved_shell = false;
        LMEAssembler asm_flat(params);
        REQUIRE_THROWS_AS(
            asm_flat.assemble_K(sphere.V, sphere.F, mat),
            std::invalid_argument);
    }
}

TEST_CASE("compute_shell_modes: LME path rejects disconnected mesh "
          "with ghosts and SME both off (review4 F4)",
          "[lme][assembler][manifold]")
{
    // The connectivity guard is reached internally only via
    // collect_boundary_edges, which is gated on use_ghost_nodes /
    // use_second_order_sme. With BOTH off the curved-LME assembler skips
    // build_edges, so before the fix a disconnected mesh reached the rigid
    // filter and returned spurious second-component ~0 Hz modes. The guard
    // now lives in the compute_shell_modes(assembler) overload, so every
    // path is covered uniformly.
    Eigen::MatrixXd V(6, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         3.0, 0.0, 0.0,
         4.0, 0.0, 0.0,
         3.0, 1.0, 0.0;
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         3, 4, 5;

    LMEAssembler::Params p;
    p.use_ghost_nodes       = false;
    p.use_second_order_sme  = false;
    LMEAssembler asm0(p);

    const chladni::IsotropicMaterial iso{2.0e11, 0.3, 7850.0};
    const ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(iso, /*thickness=*/1.0e-3);

    REQUIRE_THROWS_WITH(
        chladni::shell::compute_shell_modes(
            V, F, iso, sm, /*thickness=*/1.0e-3, /*n_modes=*/3, asm0),
        Catch::Matchers::ContainsSubstring("disconnected"));
}

TEST_CASE("compute_shell_modes: rejects out-of-range face index (review4 F5)",
          "[lme][assembler][manifold]")
{
    // A face index >= V.rows() would write out of bounds in the mass loop
    // and under-cover the rigid filter; build_edges sizes from F alone and
    // never compares against V.rows(), so the entry point must catch it.
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;
    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 3;  // vertex 3 does not exist (V.rows() == 3)

    LMEAssembler asm0;
    const chladni::IsotropicMaterial iso{2.0e11, 0.3, 7850.0};
    const ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(iso, /*thickness=*/1.0e-3);

    REQUIRE_THROWS_WITH(
        chladni::shell::compute_shell_modes(
            V, F, iso, sm, /*thickness=*/1.0e-3, /*n_modes=*/1, asm0),
        Catch::Matchers::ContainsSubstring("out of range"));
}

TEST_CASE("LMEAssembler: rejects negative drop-fraction budgets (review4 F7)",
          "[lme][assembler]")
{
    LMEAssembler::Params p;

    SECTION("negative max_newton_drop_frac")
    {
        p.max_newton_drop_frac = -0.1;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("negative max_total_drop_frac")
    {
        p.max_total_drop_frac = -0.1;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("zero and >= 1 are both accepted")
    {
        p.max_newton_drop_frac = 0.0;
        p.max_total_drop_frac  = 2.0;  // >= 1 disables the abort
        REQUIRE_NOTHROW(LMEAssembler{p});
    }
}

TEST_CASE("LMEAssembler: rejects out-of-range chart-sizing parameters (review6 H3)",
          "[lme][assembler]")
{
    // gamma, gamma_wpca, gamma_pu, tol_lme, chart_tol_lme and r_cut_mult feed
    // the gamma_x/h² rates and sqrt(-log(.)) cutoff-radius expressions in
    // build_curved_patch_context, which run before any evaluate_basis boundary
    // check. A bad value yields a NaN radius / non-positive beta (silent 2-ring
    // collapse or a late opaque Newton/PoU throw), so the ctor must reject it
    // up front. sme_alpha/sme_beta carry the stricter compute_nodal_gaps bounds
    // (alpha > 1, beta >= 1) but only when use_second_order_sme is on (review7
    // I1/I2/I3). Defaults must still construct.
    SECTION("default Params constructs")
    {
        REQUIRE_NOTHROW(LMEAssembler{LMEAssembler::Params{}});
    }
    SECTION("gamma <= 0 throws")
    {
        LMEAssembler::Params p; p.gamma = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("tol_lme >= 1 throws (-log <= 0 -> NaN radius)")
    {
        LMEAssembler::Params p; p.tol_lme = 1.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("tol_lme <= 0 throws")
    {
        LMEAssembler::Params p; p.tol_lme = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("chart_tol_lme >= 1 throws")
    {
        LMEAssembler::Params p; p.chart_tol_lme = 1.5;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("r_cut_mult <= 0 throws")
    {
        LMEAssembler::Params p; p.r_cut_mult = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("gamma_wpca <= 0 throws (review7 I3)")
    {
        LMEAssembler::Params p; p.gamma_wpca = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("gamma_pu <= 0 throws (review7 I3)")
    {
        LMEAssembler::Params p; p.gamma_pu = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    // sme_alpha / sme_beta are validated to their compute_nodal_gaps bounds
    // (alpha > 1, beta >= 1) but ONLY when use_second_order_sme is on — they
    // are not consulted otherwise (review7 I1/I2).
    SECTION("sme_alpha in (0,1] throws when SME on (review7 I1)")
    {
        LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.sme_alpha            = 0.5;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("sme_alpha == 1 throws when SME on (boundary, review7 I1)")
    {
        LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.sme_alpha            = 1.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("sme_beta < 1 throws when SME on (review7 I2)")
    {
        LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.sme_beta             = 0.5;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("sme_alpha <= 0 is ignored when SME off (not consulted)")
    {
        LMEAssembler::Params p; p.sme_alpha = 0.0;  // SME off (default)
        REQUIRE_NOTHROW(LMEAssembler{p});
    }
    SECTION("NaN gamma throws")
    {
        LMEAssembler::Params p;
        p.gamma = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    // The chart-sizing / solver-control siblings the review6/7 cluster left
    // unvalidated (review8 J3/J4/J5). max_chart_nodes <= 0 is UB in the cap
    // branch; newton_tol/newton_max_iters surface as opaque worker-thread
    // "did not converge" throws; n_quadrature_per_tri was validated only late
    // at assembly time.
    SECTION("max_chart_nodes < 7 throws (review8 J3 — UB-prone cap)")
    {
        LMEAssembler::Params p; p.max_chart_nodes = 0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("newton_tol <= 0 throws (review8 J4)")
    {
        LMEAssembler::Params p; p.newton_tol = 0.0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("newton_max_iters < 1 throws (review8 J4)")
    {
        LMEAssembler::Params p; p.newton_max_iters = 0;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
    SECTION("n_quadrature_per_tri outside {1,3,7,12} throws up front (review8 J5)")
    {
        LMEAssembler::Params p; p.n_quadrature_per_tri = 5;
        REQUIRE_THROWS_AS(LMEAssembler{p}, std::invalid_argument);
    }
}

TEST_CASE("LMEAssembler::Params: SME flags default off with paper-recommended scalars",
          "[lme][assembler][sme]")
{
    // Defaults pinned to the paper's operating range
    // (Rosolen-Millan-Arroyo 2013 §3.2/§4: alpha in [1.6, 2.5], beta ~ 1)
    // and OFF by default to preserve existing performance characteristics
    // on the curved-shell path. alpha=2 is the paper's central value;
    // the earlier 4.0 was a workaround for a since-fixed Newton
    // convergence bug (it mis-reported feasible alpha=2 solves as
    // divergences) and over-stiffened curved shells. Phase B.2's
    // regression bar (32x8 polar disk Leissa) opts in via
    // use_second_order_sme=true.
    const LMEAssembler::Params p;
    REQUIRE(p.use_second_order_sme == false);
    REQUIRE(p.sme_alpha == 2.0);
    REQUIRE(p.sme_beta  == 1.0);
}

TEST_CASE("LMEAssembler::assemble_K / assemble_M: SME requires curved-shell path",
          "[lme][assembler][sme]")
{
    // SME is wired into the per-Gauss-point evaluation of the
    // curved-shell assembler (chart-2D Newton on the SME dual);
    // the legacy flat-plate path has no per-chart wPCA frame and
    // therefore no place to plug SME in. Misuse must fail fast at
    // assemble entry, not silently fall through to the LME path.
    const auto plate = chladni::mesh::generate_flat_plate(1.0, 1.0, 3, 3);
    const ShellMaterial mat{0.0, 1.0, 0.3};

    LMEAssembler::Params params;
    params.use_curved_shell     = false;
    params.use_second_order_sme = true;
    LMEAssembler asm_bad(params);
    REQUIRE_THROWS_AS(asm_bad.assemble_K(plate.V, plate.F, mat),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(asm_bad.assemble_M(plate.V, plate.F, 1.0),
                      std::invalid_argument);
}

TEST_CASE("LMEAssembler::assemble_K / assemble_M (curved): reject non-finite "
          "input geometry (review4 F1 / review5 G10)",
          "[lme][assembler][manifold]")
{
    // F1 added an allFinite backstop to the curved mass assembler to
    // match curved-K; review5 G10 noted no test pinned the M side. A
    // non-finite vertex coordinate must not silently reach the
    // generalized eigenproblem on either the K or M curved path -- it is
    // caught at the spacing source (vertex_one_ring_h, review5 G7) or by
    // the assembled-stream allFinite backstop, but it must throw.
    auto plate = chladni::mesh::generate_flat_plate(0.1, 0.1, 4, 4);
    plate.V(0, 2) = std::numeric_limits<double>::quiet_NaN();

    const ShellMaterial mat{1.0, 1.0, 0.3};

    LMEAssembler asm0;  // default: curved-shell path on
    REQUIRE_THROWS(asm0.assemble_K(plate.V, plate.F, mat));
    REQUIRE_THROWS(asm0.assemble_M(plate.V, plate.F, 1.0));
}

TEST_CASE("LMEAssembler::label() returns paper-style identifier",
          "[lme][assembler]")
{
    LMEAssembler asm0;
    REQUIRE(asm0.label() == "LME (Arroyo-Ortiz)");
}

TEST_CASE("compute_shell_modes(LMEAssembler) end-to-end on a flat plate",
          "[lme][assembler][modes]")
{
    // Polymorphic entry-point smoke test. Drives the full
    // assemble_K + assemble_M + rigid-filter +
    // evaluate_modes_at_vertices pipeline through LMEAssembler. The
    // assembler-direct path is exercised by the SS plate / free-edge
    // disk / annular plate tests; this catches integration regressions
    // in the compute_shell_modes wiring (worker-thread dispatch in
    // strike_gui hangs off this exact call signature).
    const auto plate = chladni::mesh::generate_flat_plate(
        /*length_a=*/0.1, /*length_b=*/0.1, /*n_x=*/8, /*n_y=*/8);
    constexpr double h = 1.0e-3;
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    constexpr std::size_t n_modes = 8;
    chladni::shell::LMEAssembler assembler;

    const auto modes = chladni::shell::compute_shell_modes(
        plate.V, plate.F, steel, sm, h, n_modes, assembler);

    // Eigenvalues: count, sorted ascending, all strictly positive
    // (no rigid-body residue leaking through the 6-dim filter).
    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes));
    for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
        INFO("mode " << i << " ω = " << modes.omegas(i));
        REQUIRE(std::isfinite(modes.omegas(i)));
        REQUIRE(modes.omegas(i) > 0.0);
    }
    for (Eigen::Index i = 1; i < modes.omegas.size(); ++i) {
        REQUIRE(modes.omegas(i) >= modes.omegas(i - 1));
    }

    // Shapes: full 3·n_V × n_modes layout; each column has nontrivial
    // norm (no all-zero or all-NaN modes).
    REQUIRE(modes.shapes.rows() == 3 * plate.V.rows());
    REQUIRE(modes.shapes.cols() == static_cast<Eigen::Index>(n_modes));
    for (Eigen::Index k = 0; k < modes.shapes.cols(); ++k) {
        const double nrm = modes.shapes.col(k).norm();
        INFO("mode " << k << " ||shape||_2 = " << nrm);
        REQUIRE(std::isfinite(nrm));
        REQUIRE(nrm > 0.0);
    }
}

TEST_CASE("compute_shell_modes(LMEAssembler flat path) throws on curved input",
          "[lme][assembler][modes]")
{
    // The legacy flat-plate path (use_curved_shell=false) asserts
    // strict planarity. The default-curved path now handles curved
    // meshes natively; this test pins the flat-path behaviour only.
    const auto sphere = chladni::mesh::generate_icosphere(
        /*radius=*/1.0, /*n_subdivisions=*/2);
    constexpr double h = 1.0e-3;
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 1.0 / 3.0,
        .density        = 7850.0};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);

    chladni::shell::LMEAssembler::Params params;
    params.use_curved_shell = false;
    chladni::shell::LMEAssembler assembler(params);
    REQUIRE_THROWS_AS(
        chladni::shell::compute_shell_modes(
            sphere.V, sphere.F, steel, sm, h, /*n_modes=*/4, assembler),
        std::invalid_argument);
}
