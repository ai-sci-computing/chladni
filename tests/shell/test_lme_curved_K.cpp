/**
 * @file test_lme_curved_K.cpp
 * @brief Curved-shell LME stiffness assembly: flat-plate sanity and
 *        curved-input plumbing.
 *
 * The curved-shell path of @ref chladni::shell::LMEAssembler::assemble_K
 * (selected via @c Params::use_curved_shell = true) walks per-patch
 * tangent charts whose neighbour lists are k-ring topological
 * neighbourhoods through @c F. That localisation is what makes the
 * formulation robust on curved meshes (it stops antipodal vertices
 * from projecting onto the chart anchor and collapsing the in-chart
 * basis), but it also means the assembled K is **not** algebraically
 * equal to the legacy flat-plate path's K — the local LME bases on
 * overlapping patches are blended through the Shepard PU rather than
 * solved once globally.
 *
 * The flat-plate gate is therefore an eigenvalue match, not an
 * entrywise match: with bending-only material we extract the @c z-block
 * generalised eigenvalues of both K's and require the lowest non-rigid
 * modes to agree to within 5 %. The 5 % budget is chosen to cover the
 * Shepard-PU localisation error at the @c γ_pu = 4 default; tighter
 * @c γ_pu (more concentrated PU) recovers entrywise agreement in the
 * limit @c γ_pu → ∞ but burns the smoothness the PU buys us.
 *
 * The icosphere case asserts plumbing only — symmetric K, no throws,
 * correct DOF layout. The physical-correctness gate against
 * @ref LoopAssembler lands in 9.7.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell/lme.hpp>

#include "lme_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <limits>
#include <vector>

using chladni::shell::LMEAssembler;
using chladni::shell::ShellMaterial;
using chladni::tests::lme::solve_lme_z_modes;
using Catch::Matchers::WithinAbs;

namespace {

ShellMaterial steel_bending_only(double h_thickness)
{
    constexpr double E_steel = 2.0e11;
    constexpr double nu      = 0.30;
    ShellMaterial    m;
    m.poisson_ratio = nu;
    m.k_L           = 0.0;  // disable membrane to isolate bending
    m.k_B           = E_steel * std::pow(h_thickness, 3)
                        / (12.0 * (1.0 - nu * nu));
    return m;
}

ShellMaterial steel_full(double h_thickness)
{
    constexpr double E_steel = 2.0e11;
    constexpr double nu      = 0.30;
    ShellMaterial    m;
    m.poisson_ratio = nu;
    m.k_L           = E_steel * h_thickness / (1.0 - nu * nu);
    m.k_B           = E_steel * std::pow(h_thickness, 3)
                        / (12.0 * (1.0 - nu * nu));
    return m;
}

/// Build the 6 rigid-body modes of a 3D-displacement vertex layout
/// (3 translations + 3 rotations about the cloud's centroid). Returns
/// a (3·n_v) × 6 matrix whose columns span the rigid-body subspace.
Eigen::MatrixXd rigid_body_modes(const Eigen::MatrixXd& V)
{
    const Eigen::Index n_v = V.rows();
    Eigen::RowVector3d centroid = V.colwise().mean();
    Eigen::MatrixXd    R(3 * n_v, 6);
    R.setZero();
    for (Eigen::Index a = 0; a < n_v; ++a) {
        // Translations: T_x, T_y, T_z
        for (int k = 0; k < 3; ++k) R(3 * a + k, k) = 1.0;
        // Rotations about centroid: ω × (P_a - c) for ω = x̂, ŷ, ẑ
        const Eigen::RowVector3d r = V.row(a) - centroid;
        // R_x rotation: ω = (1,0,0) → ω × r = (0, -r_z, r_y)
        R(3 * a + 1, 3) = -r.z();  R(3 * a + 2, 3) =  r.y();
        // R_y rotation: ω = (0,1,0) → ω × r = (r_z, 0, -r_x)
        R(3 * a + 0, 4) =  r.z();  R(3 * a + 2, 4) = -r.x();
        // R_z rotation: ω = (0,0,1) → ω × r = (-r_y, r_x, 0)
        R(3 * a + 0, 5) = -r.y();  R(3 * a + 1, 5) =  r.x();
    }
    return R;
}

}  // namespace

TEST_CASE("LMEAssembler::assemble_K curved bending: eigenvalues on a flat "
          "plate agree with the legacy flat path within 5%",
          "[shell][lme][assembler][curved_K]")
{
    constexpr int    N   = 8;
    constexpr double L   = 0.01;   // 1 cm side
    constexpr double thk = 0.001;  // 1 mm thickness

    auto mesh = chladni::mesh::generate_flat_plate(L, L, N, N);
    const Eigen::MatrixXd& V = mesh.V;
    const Eigen::MatrixXi& F = mesh.F;

    const ShellMaterial material           = steel_bending_only(thk);
    constexpr double    rho_h              = 7800.0 * thk;
    std::vector<bool>   none_pinned(static_cast<std::size_t>(V.rows()), false);
    constexpr std::size_t n_compute = 6;

    LMEAssembler::Params params_flat;
    params_flat.use_curved_shell = false;
    LMEAssembler::Params params_curved;
    params_curved.use_curved_shell = true;
    // Compare the legacy curved-on-flat assembly against the flat path;
    // ghost-on extends the modelled domain beyond the plate rim, so the
    // eigenvalues would intentionally diverge — that's a separate gate.
    params_curved.use_ghost_nodes = false;
    // "Curved reduces to flat on flat input" reduction. The curved path
    // selects its active set by the paper's VALUE-based truncation
    // (Millán Eq. 2 via tol_lme), not by r_cut_mult_curved (retired/
    // inert) — at the shared default tol_lme/γ its per-node radius spans
    // a comparable active set to the flat path's r_cut_mult cutoff, so
    // the reduction holds with no pinning needed.

    const auto modes_flat = solve_lme_z_modes(
        V, F, material, rho_h, none_pinned, n_compute, params_flat);
    const auto modes_curved = solve_lme_z_modes(
        V, F, material, rho_h, none_pinned, n_compute, params_curved);

    // Skip the 3 rigid-body z-block modes (translation + 2 rotations);
    // compare the genuine bending eigenvalues only.
    for (std::size_t k = 3; k < n_compute; ++k) {
        const double lam_flat   = modes_flat.eigenvalues(
            static_cast<Eigen::Index>(k));
        const double lam_curved = modes_curved.eigenvalues(
            static_cast<Eigen::Index>(k));
        REQUIRE(lam_flat > 0.0);
        const double rel_err = std::abs(lam_curved - lam_flat) / lam_flat;
        REQUIRE(rel_err < 0.05);
    }
}

TEST_CASE("LMEAssembler::assemble_K curved bending+membrane: 6-dim rigid "
          "kernel preserved exactly on a flat plate",
          "[shell][lme][assembler][curved_K]")
{
    constexpr int    N   = 6;
    constexpr double L   = 0.01;
    constexpr double thk = 0.001;

    auto mesh = chladni::mesh::generate_flat_plate(L, L, N, N);
    const Eigen::MatrixXd& V = mesh.V;
    const Eigen::MatrixXi& F = mesh.F;

    const ShellMaterial material = steel_full(thk);

    LMEAssembler::Params params;
    params.use_curved_shell = true;
    // Rigid kernel is asserted via rigid_body_modes(V) which is sized to
    // the real-vertex count; the ghost-on path inflates K to 3·(N+G) and
    // its kernel includes ghost-position-dependent rotation modes that
    // aren't representable from V alone.
    params.use_ghost_nodes = false;

    const auto Kc = LMEAssembler(params).assemble_K(V, F, material);
    const Eigen::MatrixXd K_dense = Eigen::MatrixXd(Kc);

    // Symmetric.
    REQUIRE((K_dense - K_dense.transpose()).norm() / K_dense.norm()
            < 1.0e-10);

    // Rigid-body kernel: K · V_rigid = 0 for all 6 modes.
    const Eigen::MatrixXd Rmodes = rigid_body_modes(V);
    const Eigen::MatrixXd K_R    = K_dense * Rmodes;
    const double          K_norm = K_dense.norm();
    REQUIRE(K_norm > 0.0);
    // Numerical-tolerance budget: Newton tol 1e-10 propagated through
    // matrix mults, Shepard PoU normalisation, and ~150 patch
    // contributions per Gauss point each add ~1e-9. The combined
    // residual on linear-reproduction (= rigid-kernel preservation)
    // sits at ~1e-7 on this fixture — still ~12 orders below K_norm.
    REQUIRE(K_R.norm() / K_norm < 1.0e-6);

    // Sanity: there are physical bending+membrane modes (some non-rigid
    // displacement has positive strain energy). Pick the "u_x = x"
    // membrane stretch on the in-plane components.
    Eigen::VectorXd u_stretch(3 * V.rows());
    u_stretch.setZero();
    for (Eigen::Index a = 0; a < V.rows(); ++a) {
        u_stretch(3 * a + 0) = V(a, 0);  // u_x = x (uniform stretch)
    }
    const double e_stretch = 0.5 * u_stretch.transpose() * K_dense * u_stretch;
    REQUIRE(e_stretch > 1.0e-4);
}

TEST_CASE("LMEAssembler::assemble_K curved+SME: symmetric K on closed icosphere",
          "[shell][lme][assembler][curved_K][sme]")
{
    // Closed mesh (no boundary) — the simplest SME-on smoke: every
    // chart node classifies as Interior, the SME convex program is
    // well-conditioned for every Gauss point. Validates the
    // dispatch wiring and basic K invariants without bringing in
    // chart-2D boundary topology.
    auto mesh = chladni::mesh::generate_icosphere(/*radius=*/1.0,
                                                  /*n_subdivisions=*/1);
    const Eigen::MatrixXd& V = mesh.V;
    const Eigen::MatrixXi& F = mesh.F;

    const ShellMaterial material = steel_bending_only(0.001);

    LMEAssembler::Params params;
    params.use_curved_shell     = true;
    params.use_ghost_nodes      = false;  // closed mesh: no ghosts anyway
    params.use_second_order_sme = true;

    Eigen::SparseMatrix<double> K;
    REQUIRE_NOTHROW(K = LMEAssembler(params).assemble_K(V, F, material));
    REQUIRE(K.rows() == 3 * V.rows());
    REQUIRE(K.cols() == 3 * V.rows());

    const Eigen::MatrixXd K_dense = Eigen::MatrixXd(K);
    REQUIRE(K_dense.norm() > 0.0);
    REQUIRE((K_dense - K_dense.transpose()).norm() / K_dense.norm()
            < 1.0e-10);
}

TEST_CASE("LMEAssembler::assemble_K curved+SME: symmetric K on flat plate, ghost-on",
          "[shell][lme][assembler][curved_K][sme]")
{
    // Wire-up sanity for Phase B.1: the SME path must run on the
    // default (ghost-on) curved-shell config, producing a symmetric
    // K of the ghost-extended shape with strictly-positive bending
    // energy on a non-rigid deformation. Tighter accuracy gates
    // come from Phase B.2's Leissa regression bar.
    //
    // Mesh sized so kRing=3 charts cover only a local neighbourhood
    // (NOT the entire mesh). On a 4x4 plate every chart spans every
    // vertex; the wPCA projection becomes near-singular and SME's
    // 2nd-order moment constraint becomes infeasible in every
    // Shepard-active patch simultaneously. 12x12 keeps charts local.
    //
    // L=1.0 (not 0.01) so node spacing h≈0.08 stays in the O(1)
    // regime where SME's Newton converges robustly. With h<<1 the
    // packed PHI matrix has columns at O(h) and O(h²), and the
    // infinity-norm Newton convergence check on absolute g requires
    // ~h² relative accuracy in λ — too tight for the 50-iter cap.
    // The B.2 polar-disk fixture lives at L~O(1) anyway.
    constexpr int    N   = 12;
    constexpr double L   = 1.0;
    constexpr double thk = 0.001;

    auto mesh = chladni::mesh::generate_flat_plate(L, L, N, N);
    const Eigen::MatrixXd& V = mesh.V;
    const Eigen::MatrixXi& F = mesh.F;

    const ShellMaterial material = steel_full(thk);

    LMEAssembler::Params params;
    params.use_curved_shell     = true;
    params.use_ghost_nodes      = true;
    params.use_second_order_sme = true;
    // No r_cut override: SME truncation is VALUE-based (γ_eff = 2/α,
    // ≈4.80 h at α=2; inventory C7, r_cut_mult_sme retired) — the wide
    // active set it needs (tighter diverged — that is what crashed the
    // GUI before SME stopped sharing LME's 1.4 default).

    const auto Kc = LMEAssembler(params).assemble_K(V, F, material);
    const Eigen::MatrixXd K_dense = Eigen::MatrixXd(Kc);

    REQUIRE(K_dense.rows() > 3 * V.rows());  // ghost-extended
    REQUIRE(K_dense.cols() == K_dense.rows());

    REQUIRE((K_dense - K_dense.transpose()).norm() / K_dense.norm()
            < 1.0e-10);

    // Real-vertex z-stretch (out-of-plane deformation) has non-zero
    // bending energy. Picks a quadratic profile so the deformation
    // is genuinely non-rigid (rigid modes are linear) without
    // assuming any particular eigenvector.
    const Eigen::Index n_dof = K_dense.rows();
    Eigen::VectorXd u_quad(n_dof);
    u_quad.setZero();
    for (Eigen::Index a = 0; a < V.rows(); ++a) {
        u_quad(3 * a + 2) = V(a, 0) * V(a, 0) + V(a, 1) * V(a, 1);
    }
    const double e_quad = 0.5 * u_quad.transpose() * K_dense * u_quad;
    REQUIRE(e_quad > 0.0);
}

TEST_CASE("LMEAssembler curved+SME: out-of-range sme_alpha rejected at the ctor "
          "(review6 H3 / review7 I1)",
          "[shell][lme][assembler][curved_K][sme]")
{
    // VEHICLE HISTORY: the GUI-crash regression below originally
    // provoked a deep SME throw via a too-tight r_cut_mult_sme = 1.4
    // (retired 2026-06-04, C7), then via NEAR-ZERO SLACK sme_alpha ≈ 0.
    // review7 I1 closed that second vehicle: sme_alpha <= 1 with SME on
    // is now rejected UP FRONT at the ctor (compute_nodal_gaps requires
    // alpha > 1), so the bad value can no longer reach assembly and
    // throw late/opaquely on a worker thread. Pin that early rejection.
    LMEAssembler::Params params;
    params.use_curved_shell     = true;
    params.use_ghost_nodes      = true;
    params.use_second_order_sme = true;
    params.sme_alpha            = 0.05;  // < 1: now a ctor-level error
    REQUIRE_THROWS_AS(LMEAssembler{params}, std::invalid_argument);
}

TEST_CASE("LMEAssembler::assemble_K curved+SME: an in-worker throw is catchable, "
          "not std::terminate",
          "[shell][lme][assembler][curved_K][sme]")
{
    // Regression for the GUI crash (2026-05-27): a per-patch assembly
    // failure on the curved+SME path throws from inside a worker
    // std::thread; before run_threaded captured + rethrew worker
    // exceptions it escaped as std::terminate — a hard SIGABRT the GUI's
    // try/catch could not intercept, so toggling SME crashed the process
    // instantly. It must surface as a normal catchable std::exception.
    //
    // VEHICLE: the original infeasibility triggers are gone — r_cut_mult_sme
    // = 1.4 retired (C7, 2026-06-04), the SME cylinder divergence was cured
    // (2026-06-03), and a too-small sme_alpha is now rejected at the ctor
    // (review7 I1). What remains, and is what the rethrow genuinely guards,
    // is a per-triangle worker throw: a non-finite vertex collapses the
    // first-fundamental-form metric, which the curved kernel detects and
    // throws on at the quadrature point (lme.cpp degenerate-metric guard),
    // INSIDE the worker. With valid SME params this exercises the
    // run_threaded capture/rethrow exactly as the GUI hit it.
    auto mesh = chladni::mesh::generate_circular_disk(0.1, 32, 8);
    mesh.V(0, 2) = std::numeric_limits<double>::quiet_NaN();
    const ShellMaterial material = steel_full(0.001);

    LMEAssembler::Params params;
    params.use_curved_shell     = true;
    params.use_ghost_nodes      = true;
    params.use_second_order_sme = true;
    params.sme_alpha            = 2.0;  // valid: passes the ctor

    REQUIRE_THROWS_AS(
        LMEAssembler(params).assemble_K(mesh.V, mesh.F, material),
        std::exception);
}

TEST_CASE("LMEAssembler::assemble_K curved path: runs on icosphere, "
          "produces symmetric K with correct shape",
          "[shell][lme][assembler][curved_K]")
{
    auto mesh = chladni::mesh::generate_icosphere(/*radius=*/1.0,
                                                  /*n_subdivisions=*/2);
    const Eigen::MatrixXd& V = mesh.V;
    const Eigen::MatrixXi& F = mesh.F;

    const ShellMaterial material = steel_bending_only(0.001);

    LMEAssembler::Params params;
    params.use_curved_shell = true;

    Eigen::SparseMatrix<double> K;
    REQUIRE_NOTHROW(K = LMEAssembler(params).assemble_K(V, F, material));
    REQUIRE(K.rows() == 3 * V.rows());
    REQUIRE(K.cols() == 3 * V.rows());

    const Eigen::MatrixXd K_dense = Eigen::MatrixXd(K);
    REQUIRE(K_dense.norm() > 0.0);
    REQUIRE((K_dense - K_dense.transpose()).norm() / K_dense.norm()
            < 1.0e-10);
}
