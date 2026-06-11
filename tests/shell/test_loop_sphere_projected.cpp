/**
 * @file test_loop_sphere_projected.cpp
 * @brief Convergence and validation of the Loop shell FEM against
 *        Wilkinson on a sequence of finer sphere discretisations.
 *
 * Historical note: an earlier version of this file (and the
 * "icosphere gap diagnosis" in commit 6237bff) attributed the 36 %
 * FEM-vs-Wilkinson rel_err at k=2 to **geometric** Loop-limit-surface
 * shrinkage. That diagnosis was wrong — the dominant effect was the
 * @ref chladni::shell::MassLumping::RowSum mass-lumping default
 * shipped at the time, which lowers eigenfrequencies broadly across
 * mesh types. The default was flipped to @ref MassLumping::None on
 * 2026-05-17 (late); see the @c [.experiment] Mass-lumping probe
 * below for the data that pinned this.
 *
 * @section sequence Clean convergence sequence (consistent mass default)
 *
 * The canonical test is icosphere(k=2, 3, 4, ...) — every vertex on
 * the sphere by construction, no projection trick mid-flight.
 * Measured rel_err (n=2 cluster mean vs Wilkinson) under the
 * consistent-mass default:
 *
 *     k=2: L.3.4 0.93 %, Stam 2.26 %
 *     k=3: L.3.4 0.26 %, Stam 0.32 %
 *     k=4: L.3.4 0.044 %, Stam 0.064 %
 *
 * Convergence ratios (rel_err(k+1) / rel_err(k)): L.3.4 ≈ 0.28
 * (k=2→3) and 0.17 (k=3→4); Stam ≈ 0.14 and 0.20. Both paths
 * converge with the expected O(h²)-ish trajectory; the absolute
 * level at k=2 is already sub-percent on L.3.4, so the "Stam beats
 * L.3.4 by 4×" reading from the pre-flip era no longer applies.
 *
 * The genuine geometric Loop-limit-surface effect (the limit surface
 * sits ~2 % inside the target sphere with ~0.5 % lumpiness near
 * valence-5 vertices) is visible in the convergence rate and the
 * small but real Stam-vs-L.3.4 spread at k=2, but it is dwarfed by
 * the lumping bias that previously dominated.
 *
 * @section validation Regression test
 *
 * Raw icosphere(k=4) L.3.4 → rel_err < 0.5 % pins the FEM pipeline
 * end-to-end on a closed-shell fixture, comparable in tightness to
 * the cylinder and flat-plate analytic validations.
 *
 * @section projection Projection trick — kept as a curiosity, not a fix
 *
 * Earlier in the investigation we explored projecting the
 * Loop-subdivided icosphere back onto the sphere of radius R. It
 * does close the gap at low k, but only because it's equivalent to
 * doing one more refinement step — the projected k=3 mesh has
 * 2562 vertices on the sphere, essentially the same shape regime
 * as raw icosphere(k=4). Kept in the [.experiment] sweep for
 * documentation.
 */

#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/contrib/LOBPCGSolver.h>

#include <Eigen/SparseLU>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

chladni::IsotropicMaterial steel()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.30,
            .density        = 7850.0};
}

// Project every vertex of @p V onto the sphere of radius @p R centred at
// the origin. Operates in place.
void project_to_sphere(Eigen::MatrixXd& V, double R)
{
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        const double r = V.row(i).norm();
        if (r > 0.0) V.row(i) *= (R / r);
    }
}

double n2_mean_hz_for_icosphere(int k_init, bool project, int n_passes,
                                bool use_stam)
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;

    auto mesh = cmsh::generate_icosphere(R, k_init);

    if (project) {
        // Subdivide once, project the subdivided mesh back to the
        // sphere, then run the FEM on the projected mesh. The Loop
        // pipeline will subdivide AGAIN internally; the limit surface
        // is the second-generation limit of a control mesh whose
        // vertices all sit exactly on the sphere.
        const auto sub = csl::loop_subdivide_one_step(mesh.V, mesh.F);
        Eigen::MatrixXd V_proj = sub.V_sub;
        project_to_sphere(V_proj, R);
        const auto modes = cs::compute_shell_modes_loop(
            V_proj, sub.F_sub, steel(), h,
            /*n_modes=*/6, n_passes, use_stam);
        REQUIRE(modes.omegas.allFinite());
        double sum = 0.0;
        for (Eigen::Index i = 0; i < 5; ++i) sum += modes.omegas(i);
        return (sum / 5.0) / (2.0 * std::numbers::pi);
    }

    const auto modes = cs::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h,
        /*n_modes=*/6, n_passes, use_stam);
    REQUIRE(modes.omegas.allFinite());
    double sum = 0.0;
    for (Eigen::Index i = 0; i < 5; ++i) sum += modes.omegas(i);
    return (sum / 5.0) / (2.0 * std::numbers::pi);
}

double wilkinson_n2_hz()
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto analytical =
        chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel(), /*n_modes=*/1);
    return analytical.front() / (2.0 * std::numbers::pi);
}

}  // namespace

TEST_CASE("Convergence on raw icosphere(k=2..5): L.3.4 vs Stam",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Methodologically clean version: every vertex of icosphere(k) is
    // exactly on the sphere of radius R by generate_icosphere's
    // construction (each subdivision step inserts edge midpoints and
    // re-projects them to R, the existing vertices are NOT touched).
    // So this sequence — k=2..5 — is a series of finer and finer
    // sphere discretizations with no projection trick mid-flight.
    //
    // We measure rel_err vs Wilkinson at each k, plus the convergence
    // ratio per refinement step, on both algorithms (L.3.4 default
    // path and Stam exact-eval). The k=5 extension (10242 V,
    // ~30k DOFs) was added to discriminate the geometric-limit-surface
    // hypothesis from a possible Stam-quadrature-bias hypothesis on
    // the small but stable Stam-vs-L.3.4 gap captured in
    // chladni-stam-vs-l34-consistent-mass — if the gap narrows toward
    // ~0.005 pp at k=5, it's geometric; if it stays ~0.02 pp, it
    // points to a Stam bias.
    const double f_w = wilkinson_n2_hz();
    std::cout << "[icosphere RAW sequence — Wilkinson n=2 = "
              << f_w << " Hz]\n";

    struct Result {
        int k;
        bool use_stam;
        std::size_t n_vertices;
        double f_n2_mean;
        double rel_err;
    };
    std::vector<Result> rows;
    for (int k : {2, 3, 4, 5}) {
        for (bool use_stam : {false, true}) {
            const auto mesh = cmsh::generate_icosphere(0.10, k);
            const double f = n2_mean_hz_for_icosphere(
                k, /*project=*/false, /*n_passes=*/1, use_stam);
            rows.push_back({
                k, use_stam, static_cast<std::size_t>(mesh.V.rows()),
                f, std::abs(f - f_w) / f_w
            });
            std::cout << "  k=" << k
                      << "  V=" << mesh.V.rows()
                      << "  " << (use_stam ? "Stam " : "L.3.4")
                      << "  f_n2_mean=" << f
                      << "  rel_err=" << rows.back().rel_err << '\n';
        }
    }

    // Convergence ratios per algorithm (k→k+1). A ratio < 1 means
    // FEM is converging toward Wilkinson with refinement. Geometric
    // convergence has ratio < 0.7 per Cirak-Ortiz analysis.
    auto ratio = [&](int k, bool use_stam) {
        auto find = [&](int k_, bool s) {
            for (const auto& r : rows) {
                if (r.k == k_ && r.use_stam == s) return r.rel_err;
            }
            return 0.0;
        };
        return find(k + 1, use_stam) / find(k, use_stam);
    };
    std::cout << "  Convergence ratios (rel_err(k+1) / rel_err(k)):\n"
              << "    L.3.4: 2->3 = " << ratio(2, false)
              << ",  3->4 = " << ratio(3, false) << '\n'
              << "    Stam : 2->3 = " << ratio(2, true)
              << ",  3->4 = " << ratio(3, true) << '\n';

    // Compare Stam vs L.3.4 at each k. If S.8 was right (Stam ≈ L.3.4
    // on raw icospheres), these ratios should be ~1.0 across all k.
    // If projection-test was right (Stam wins when geometry is
    // accurate), the ratio should drop below 1.0 as k increases.
    std::cout << "  Stam / L.3.4 rel_err ratio at each k:\n";
    for (int k : {2, 3, 4, 5}) {
        double err_l34 = 0.0, err_stam = 0.0;
        for (const auto& r : rows) {
            if (r.k == k) {
                (r.use_stam ? err_stam : err_l34) = r.rel_err;
            }
        }
        std::cout << "    k=" << k << ": " << (err_stam / err_l34) << '\n';
    }
}

TEST_CASE("Stam vs L.3.4 K- and M-difference vertex-energy distribution on icosphere k=2..5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    constexpr double h_thk = 1.0e-3;
    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    for (int k : {2, 3, 4, 5}) {
        const auto mesh = cmsh::generate_icosphere(0.10, k);
        const Eigen::Index n_v = mesh.V.rows();

        auto build = [&](bool use_stam, bool which_K) {
            cs::LoopAssembler::Params p;
            p.use_stam = use_stam;
            p.m_lump   = cs::MassLumping::None;
            const cs::LoopAssembler asm_(p);
            return which_K
                ? asm_.assemble_K(mesh.V, mesh.F, sm)
                : asm_.assemble_M(mesh.V, mesh.F, rho_h);
        };
        const auto K_l34_sp  = build(false, true);
        const auto K_stam_sp = build(true,  true);
        const auto M_l34_sp  = build(false, false);
        const auto M_stam_sp = build(true,  false);
        auto stats = [&](const Eigen::SparseMatrix<double>& sp,
                         const Eigen::SparseMatrix<double>& ref) {
            Eigen::VectorXd vertex_d = Eigen::VectorXd::Zero(n_v);
            for (Eigen::Index col = 0; col < sp.outerSize(); ++col) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(sp, col);
                     it; ++it) {
                    const Eigen::Index row = it.row();
                    vertex_d(row / 3) += it.value() * it.value();
                }
            }
            for (Eigen::Index v = 0; v < n_v; ++v) {
                vertex_d(v) = std::sqrt(vertex_d(v));
            }
            constexpr Eigen::Index N_v5 = 12;
            double v5_mean = 0.0;
            for (Eigen::Index v = 0; v < N_v5; ++v) v5_mean += vertex_d(v);
            v5_mean /= static_cast<double>(N_v5);
            double v5_var = 0.0;
            for (Eigen::Index v = 0; v < N_v5; ++v) {
                const double dx = vertex_d(v) - v5_mean;
                v5_var += dx * dx;
            }
            v5_var /= static_cast<double>(N_v5);
            return std::make_tuple(sp.norm() / ref.norm(), v5_mean,
                                   std::sqrt(v5_var)
                                   / std::max(v5_mean, 1e-30));
        };
        const Eigen::SparseMatrix<double> dK = K_stam_sp - K_l34_sp;
        const Eigen::SparseMatrix<double> dM = M_stam_sp - M_l34_sp;
        const auto [ratK, v5K, stdK] = stats(dK, K_l34_sp);
        const auto [ratM, v5M, stdM] = stats(dM, M_l34_sp);

        std::cout << "  k=" << k << "  V=" << n_v
                  << "  K: ||dK||/||K||=" << ratK
                  << " v5_mean=" << v5K << " v5_relstd=" << stdK << '\n'
                  << "             M: ||dM||/||M||=" << ratM
                  << " v5_mean=" << v5M << " v5_relstd=" << stdM << '\n';
    }
}

TEST_CASE("Stam dense eigensolver cross-check at icosphere k=4",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 4. Solve the Stam (K, M) generalized eigenproblem at
    // k=4 with Eigen's dense GeneralizedSelfAdjointEigenSolver,
    // bypassing Spectra. If the spurious sub-pentet mode (which at
    // k=4 sits around 5850 Hz) disappears under the dense solver,
    // Spectra is the culprit. If it persists, the bug is elsewhere.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 4);
    const Eigen::Index n_dof = 3 * mesh.V.rows();
    std::cout << "[dense solver, k=4 Stam, V=" << mesh.V.rows()
              << "  3V=" << n_dof << "]\n";

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);
    const Eigen::MatrixXd K_d(K_sp);
    const Eigen::MatrixXd M_d(M_sp);
    REQUIRE(K_d.rows() == n_dof);
    REQUIRE(M_d.rows() == n_dof);

    // Symmetrise (sparse storage can leave tiny asymmetry from
    // accumulation order).
    Eigen::MatrixXd Ks = 0.5 * (K_d + K_d.transpose());
    Eigen::MatrixXd Ms = 0.5 * (M_d + M_d.transpose());

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        Ks, Ms, Eigen::ComputeEigenvectors);
    REQUIRE(ges.info() == Eigen::Success);
    const Eigen::VectorXd lambdas = ges.eigenvalues();
    std::cout << "  smallest 12 lambdas (eigenvalues of K phi = lambda M phi):\n   ";
    for (int i = 0; i < 12 && i < lambdas.size(); ++i) {
        std::cout << " " << lambdas(i);
    }
    std::cout << "\n  corresponding omegas (rad/s):\n   ";
    for (int i = 0; i < 12 && i < lambdas.size(); ++i) {
        // Guard against tiny negative lambdas from FP slop.
        const double lam = std::max(0.0, lambdas(i));
        std::cout << " " << std::sqrt(lam);
    }
    std::cout << "\n  corresponding f (Hz):\n   ";
    for (int i = 0; i < 12 && i < lambdas.size(); ++i) {
        const double lam = std::max(0.0, lambdas(i));
        std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';
    // Note: the smallest 6 should be ~0 (RBMs), modes 7..11 should be
    // the n=2 pentet. If the dense solver gives a clean 5-fold pentet
    // (modes 7..11 all near 5905), Spectra is the problem. If it
    // gives 4 + 1 split, the bug is in K/M after all (which probes
    // 1-3 already rule out) or is in the RBM filter (probe 4 doesn't
    // touch the filter, so a 4+1 result would point there).
}

TEST_CASE("Stam LOBPCG vs Spectra shift-invert at icosphere k=4",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 5. The Stam path at icosphere k=4 returns a 4+1 split in
    // Spectra's shift-invert eigensolver (4 modes at 5911.68 Hz, 1
    // ghost at ~5850 Hz). Dense Eigen confirms the true spectrum is
    // a clean 5-fold pentet at 5911.68 Hz, so the bug is in the
    // eigensolver, not the K/M assembly.
    //
    // Spectra is a modern C++ re-implementation of ARPACK's
    // Implicitly Restarted Arnoldi (IRA). Both lose Lanczos
    // orthogonality inside degenerate clusters, so swapping Spectra
    // for ARPACK does NOT categorically fix the bug. The algorithms
    // that DO categorically handle degenerate clusters are
    // block-iteration ones: LOBPCG, block Lanczos, Jacobi-Davidson.
    //
    // Spectra/contrib/LOBPCGSolver implements LOBPCG (Knyazev 2001)
    // with constraints — pass the 6-dim rigid-body subspace as
    // constraints and LOBPCG works in its M-orthogonal complement,
    // so it should find the lowest n_modes PHYSICAL modes with the
    // pentet resolved cleanly.
    //
    // This probe runs LOBPCG on the same Stam k=4 K/M as the dense
    // cross-check above. Expected outcome (if LOBPCG works): 5
    // frequencies near 5911.68 Hz with spread < 0.1%.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 4);
    const Eigen::Index n_v = mesh.V.rows();
    const Eigen::Index n_dof = 3 * n_v;
    std::cout << "[LOBPCG probe, k=4 Stam, V=" << n_v
              << "  3V=" << n_dof << "]\n";

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    Eigen::SparseMatrix<double> K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    Eigen::SparseMatrix<double> M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    // Build the 6-dim rigid-body subspace V_rigid (3 translations +
    // 3 rotations) — same construction as the production
    // mass-projection filter in shell.cpp.
    Eigen::MatrixXd V_rigid_dense = Eigen::MatrixXd::Zero(n_dof, 6);
    for (Eigen::Index i = 0; i < n_v; ++i) {
        const double x = mesh.V(i, 0);
        const double y = mesh.V(i, 1);
        const double z = mesh.V(i, 2);
        V_rigid_dense(3*i + 0, 0) = 1.0;
        V_rigid_dense(3*i + 1, 1) = 1.0;
        V_rigid_dense(3*i + 2, 2) = 1.0;
        V_rigid_dense(3*i + 1, 3) = -z;
        V_rigid_dense(3*i + 2, 3) =  y;
        V_rigid_dense(3*i + 0, 4) =  z;
        V_rigid_dense(3*i + 2, 4) = -x;
        V_rigid_dense(3*i + 0, 5) = -y;
        V_rigid_dense(3*i + 1, 5) =  x;
    }
    Eigen::SparseMatrix<double> V_rigid = V_rigid_dense.sparseView();

    // Initial guess X: random sparse n_dof × n_modes_block. LOBPCG
    // needs block size ≥ #modes we want to resolve. We want the
    // 5-fold pentet plus a few neighbours = 8. Spectra's
    // contrib/LOBPCG inner SymGEigsSolver hard-codes ncv≤10 (line 456
    // of LOBPCGSolver.h), so the block size is capped at 8.
    constexpr int kBlock = 8;
    std::mt19937 rng(0xC1AD61);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    Eigen::MatrixXd X_dense(n_dof, kBlock);
    for (Eigen::Index i = 0; i < n_dof; ++i) {
        for (int j = 0; j < kBlock; ++j) {
            X_dense(i, j) = uni(rng);
        }
    }
    Eigen::SparseMatrix<double> X_sp = X_dense.sparseView();

    Spectra::LOBPCGSolver<double> lobpcg(K_sp, X_sp);
    lobpcg.setB(M_sp);
    lobpcg.setConstraints(V_rigid);

    // Diagonal Jacobi preconditioner: T ≈ diag(K)^-1. Standard
    // cheapest-possible preconditioner for LOBPCG. Without it,
    // LOBPCG reduces to power iteration on a poorly-conditioned
    // generalised eigenproblem.
    Eigen::SparseMatrix<double> precond(n_dof, n_dof);
    {
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(static_cast<std::size_t>(n_dof));
        for (Eigen::Index i = 0; i < n_dof; ++i) {
            const double d = K_sp.coeff(i, i);
            trip.emplace_back(i, i, d > 0.0 ? 1.0 / d : 1.0);
        }
        precond.setFromTriplets(trip.begin(), trip.end());
    }
    lobpcg.setPreconditioner(precond);
    lobpcg.compute(/*maxit=*/2000, /*tol_div_n=*/1.0e-9);

    std::cout << "  LOBPCG info = " << lobpcg.m_info
              << " (Eigen::Success = " << Eigen::Success << ")\n"
              << "  block size = " << kBlock << '\n';

    Eigen::VectorXd evals = lobpcg.m_evalues;
    std::cout << "  LOBPCG eigenvalues (" << evals.size() << "):\n   ";
    for (int i = 0; i < evals.size(); ++i) {
        std::cout << " " << evals(i);
    }
    std::cout << "\n  Corresponding f (Hz):\n   ";
    for (int i = 0; i < evals.size(); ++i) {
        const double lam = std::max(0.0, evals(i));
        std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    std::cout << "\n  Reference (dense): clean 5-fold pentet at 5911.68 Hz\n";
}

TEST_CASE("Rayleigh-Ritz refinement of Spectra output at icosphere k=5 Stam",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 7. Spectra's ShiftInvert returns a 4+1 split at k=5 Stam
    // (4 modes at ~5912 Hz, 1 ghost at ~5617 Hz). The dense solver
    // confirms the true spectrum is 5×5912 Hz (clean pentet). If
    // Spectra captures a subspace that SPANS the true 5-dim
    // invariant subspace plus contamination from one Lanczos basis
    // vector, then Rayleigh-Ritz projection should heal it: project
    // K and M onto the captured subspace, solve the small dense
    // generalized eigenproblem on K' = V^T K V and M' = V^T M V, and
    // the resulting Ritz values are the best polynomial
    // approximations to the true eigenvalues within span(V).
    //
    // Implementation runs the production eigensolve via
    // compute_shell_modes_loop to mirror exactly what users see, then
    // does a second pass of dense Rayleigh-Ritz on the (n_modes ×
    // n_modes) projected problem. Prints both the raw and refined
    // spectra so we can see if RR heals.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    std::cout << "[Rayleigh-Ritz probe, k=5 Stam, V=" << n_v << "]\n";

    // Request many more modes than the 12 we'll display, so the
    // captured subspace has buffer beyond the pentet. If the 5th
    // pentet direction is captured somewhere in the broader set,
    // Rayleigh-Ritz can re-mix to find it.
    constexpr std::size_t n_modes_request = 50;
    const auto modes = cs::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h_thk,
        n_modes_request, /*n_passes=*/1, /*use_stam=*/true);
    REQUIRE(modes.omegas.allFinite());
    std::cout << "  Spectra ShiftInvert (production path), first 16 f (Hz):\n   ";
    for (Eigen::Index i = 0; i < 16 && i < modes.omegas.size(); ++i) {
        std::cout << " " << modes.omegas(i) / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';

    // Re-assemble K and M (compute_shell_modes_loop builds and
    // discards them; we need them for the RR projection).
    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    // V = the Spectra eigenvectors (3V × n_modes_request). Build the
    // small dense projected matrices.
    const Eigen::MatrixXd& V_modes = modes.shapes;
    const Eigen::MatrixXd K_proj = V_modes.transpose() * (K_sp * V_modes);
    const Eigen::MatrixXd M_proj = V_modes.transpose() * (M_sp * V_modes);
    std::cout << "  K_proj symmetry residual: "
              << (K_proj - K_proj.transpose()).norm() / K_proj.norm() << '\n'
              << "  M_proj symmetry residual: "
              << (M_proj - M_proj.transpose()).norm() / M_proj.norm() << '\n';

    Eigen::MatrixXd K_sym = 0.5 * (K_proj + K_proj.transpose());
    Eigen::MatrixXd M_sym = 0.5 * (M_proj + M_proj.transpose());

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        K_sym, M_sym, Eigen::ComputeEigenvectors);
    REQUIRE(ges.info() == Eigen::Success);
    Eigen::VectorXd refined_lams = ges.eigenvalues();
    std::cout << "  Rayleigh-Ritz refined f (Hz), first 16:\n   ";
    for (Eigen::Index i = 0; i < 16 && i < refined_lams.size(); ++i) {
        const double lam = std::max(0.0, refined_lams(i));
        std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';
}

namespace {
// Custom OpType for SymEigsSolver that performs y = (K - σM)⁻¹ M x
// using a pre-factored SparseLU. set_shift is a no-op so repeated
// solver construction with the same Op does NOT re-factorize.
struct SharedShiftInvertOp {
    using Scalar = double;
    const Eigen::SparseLU<Eigen::SparseMatrix<double>>* lu;
    const Eigen::SparseMatrix<double>*                   M;
    Eigen::Index n_rows;

    Eigen::Index rows() const { return n_rows; }
    Eigen::Index cols() const { return n_rows; }

    void set_shift(Scalar /*sigma*/) const {
        // intentionally empty: the wrapped LU is pre-factored.
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in, n_rows);
        Eigen::Map<Eigen::VectorXd>       y(y_out, n_rows);
        const Eigen::VectorXd Mx = (*M) * x;
        y.noalias() = lu->solve(Mx);
    }
};
}  // namespace

TEST_CASE("Shared-factorization multi-seed heals the pentet at k=5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 14. The naive multi-seed approach was proven to heal
    // the pentet (probe 11). The shared-factorization variant uses
    // the same algorithm with a pre-factored (K - σM) — verify
    // the heal still happens (no algorithmic regression from the
    // shared-factor optimization).
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    constexpr Eigen::Index nev_request = 24;

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_dof = 3 * mesh.V.rows();
    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    Eigen::SparseMatrix<double> K_shifted = K_sp - kSigma * M_sp;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(K_shifted);
    REQUIRE(lu.info() == Eigen::Success);
    SharedShiftInvertOp shared_op{&lu, &M_sp, n_dof};
    const Eigen::Index ncv =
        std::min<Eigen::Index>(4 * nev_request + 1, K_sp.rows());

    std::vector<Eigen::MatrixXd> V_runs;
    const uint32_t seeds[] = {0u, 1u, 0x3039u};
    for (uint32_t seed : seeds) {
        Spectra::SymEigsSolver<SharedShiftInvertOp>
            solver(shared_op, nev_request, ncv);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        Eigen::VectorXd init_vec(n_dof);
        for (Eigen::Index i = 0; i < n_dof; ++i) init_vec(i) = uni(rng);
        solver.init(init_vec.data());
        solver.compute(Spectra::SortRule::LargestMagn,
                       /*maxit=*/1000, /*tol=*/1.0e-10);
        REQUIRE(solver.info() == Spectra::CompInfo::Successful);
        V_runs.push_back(solver.eigenvectors());
    }
    Eigen::Index total_cols = 0;
    for (const auto& V : V_runs) total_cols += V.cols();
    Eigen::MatrixXd Combined(n_dof, total_cols);
    Eigen::Index off = 0;
    for (const auto& V : V_runs) {
        Combined.middleCols(off, V.cols()) = V;
        off += V.cols();
    }
    Eigen::MatrixXd Q(n_dof, Combined.cols());
    Eigen::Index q_cols = 0;
    for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
        Eigen::VectorXd v = Combined.col(j);
        // Two-pass modified Gram-Schmidt in the M-inner-product.
        // Second pass cleans residual non-orthogonality from
        // numerical drift in the first pass — essential when input
        // vectors come from SymEigsSolver with Euclidean norm.
        for (int pass = 0; pass < 2; ++pass) {
            for (Eigen::Index kk = 0; kk < q_cols; ++kk) {
                const double c = (Q.col(kk).transpose() * (M_sp * v))(0);
                v -= c * Q.col(kk);
            }
        }
        const double m_norm_sq = (v.transpose() * (M_sp * v))(0);
        if (m_norm_sq > 1.0e-20) {
            Q.col(q_cols) = v / std::sqrt(m_norm_sq);
            ++q_cols;
        }
    }
    Q.conservativeResize(Eigen::NoChange, q_cols);
    Eigen::MatrixXd K_proj_raw = Q.transpose() * (K_sp * Q);
    Eigen::MatrixXd M_proj_raw = Q.transpose() * (M_sp * Q);
    Eigen::MatrixXd K_proj = 0.5 * (K_proj_raw + K_proj_raw.transpose());
    Eigen::MatrixXd M_proj = 0.5 * (M_proj_raw + M_proj_raw.transpose());
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        K_proj, M_proj);
    REQUIRE(ges.info() == Eigen::Success);
    const Eigen::VectorXd refined = ges.eigenvalues();

    // Pentet members live at indices 6..10 after the 6 RBMs.
    REQUIRE(refined.size() >= 12);
    Eigen::VectorXd pentet(5);
    for (int i = 0; i < 5; ++i) {
        const double lam = std::max(0.0, refined(6 + i));
        pentet(i) = std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    const double mean = pentet.mean();
    const double spread = (pentet.array() - mean).abs().maxCoeff();
    std::cout << "  pentet members (Hz):";
    for (int i = 0; i < 5; ++i) std::cout << " " << pentet(i);
    std::cout << "\n  pentet_mean=" << mean
              << "  rel_spread=" << (spread / mean) << '\n';
    // Threshold: the canonical M-aware multi-seed in probe 11
    // (SymGEigsShiftSolver, line 919) achieves rel_spread ~3e-15.
    // This shared-factor variant uses SymEigsSolver + manual
    // M-Gram-Schmidt, whose residual non-orthogonality empirically
    // bottoms out near 3e-5 on this fixture (vs ~1e-2 spread when
    // the heal fails). 1e-4 distinguishes "heal is working"
    // (~3e-5) from "ghost mode resurfaces" (~1e-2) with 3x margin.
    // The shared-factor approach is not equivalent to the canonical
    // in precision — the test name's "heal still happens" claim is
    // qualitative; the quantitative regression bar is the canonical
    // test's machine-epsilon assertion.
    REQUIRE(spread / mean < 1.0e-4);
}

TEST_CASE("Timing: shared-factorization multi-seed across k=2..5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 13. The naive multi-seed approach refactorizes (K - σM)
    // three times (probe 12 measured a 3.2× cost). With a shared
    // factorization the cost should drop to roughly 1 factor + 3
    // Lanczos passes ≈ 1.5× single-Spectra. This probe measures the
    // realised ratio with a custom SharedShiftInvertOp.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    constexpr Eigen::Index nev_request = 24;

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    std::cout << "[timing single vs shared-fact multi-seed]  nev="
              << nev_request << '\n' << std::fixed << std::setprecision(3);

    for (int k : {2, 3, 4, 5}) {
        const auto mesh = cmsh::generate_icosphere(R, k);
        const Eigen::Index n_v = mesh.V.rows();
        const Eigen::Index n_dof = 3 * n_v;
        cs::LoopAssembler::Params p;
        p.use_stam = true;
        p.m_lump   = cs::MassLumping::None;
        const cs::LoopAssembler asm_(p);
        const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

        auto t_now = []() { return std::chrono::high_resolution_clock::now(); };
        auto t_ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // Baseline: single Spectra ShiftInvert run.
        const auto t0 = t_now();
        using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
        using OpM_  = Spectra::SparseSymMatProd<double>;
        OpKM op_km(K_sp, M_sp);
        OpM_ op_m_baseline(M_sp);
        const Eigen::Index ncv =
            std::min<Eigen::Index>(4 * nev_request + 1, K_sp.rows());
        Spectra::SymGEigsShiftSolver<OpKM, OpM_, Spectra::GEigsMode::ShiftInvert>
            solver_single(op_km, op_m_baseline, nev_request, ncv, kSigma);
        solver_single.init();
        solver_single.compute(Spectra::SortRule::LargestMagn,
                              /*maxit=*/1000, /*tol=*/1.0e-10);
        REQUIRE(solver_single.info() == Spectra::CompInfo::Successful);
        const auto t1 = t_now();
        const double dt_single = t_ms(t0, t1);

        // Shared-factor multi-seed: factor once, 3 Lanczos passes.
        const auto t2 = t_now();
        Eigen::SparseMatrix<double> K_shifted = K_sp - kSigma * M_sp;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(K_shifted);
        REQUIRE(lu.info() == Eigen::Success);
        SharedShiftInvertOp shared_op{&lu, &M_sp, n_dof};

        std::vector<Eigen::MatrixXd> V_runs;
        const uint32_t seeds[] = {0u, 1u, 0x3039u};
        for (uint32_t seed : seeds) {
            Spectra::SymEigsSolver<SharedShiftInvertOp>
                solver(shared_op, nev_request, ncv);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(-0.5, 0.5);
            Eigen::VectorXd init_vec(n_dof);
            for (Eigen::Index i = 0; i < n_dof; ++i) init_vec(i) = uni(rng);
            solver.init(init_vec.data());
            solver.compute(Spectra::SortRule::LargestMagn,
                           /*maxit=*/1000, /*tol=*/1.0e-10);
            REQUIRE(solver.info() == Spectra::CompInfo::Successful);
            V_runs.push_back(solver.eigenvectors());
        }
        Eigen::Index total_cols = 0;
        for (const auto& V : V_runs) total_cols += V.cols();
        Eigen::MatrixXd Combined(n_dof, total_cols);
        Eigen::Index off = 0;
        for (const auto& V : V_runs) {
            Combined.middleCols(off, V.cols()) = V;
            off += V.cols();
        }
        Eigen::MatrixXd Q(n_dof, Combined.cols());
        Eigen::Index q_cols = 0;
        for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
            Eigen::VectorXd v = Combined.col(j);
            for (Eigen::Index kk = 0; kk < q_cols; ++kk) {
                const double c = (Q.col(kk).transpose() * (M_sp * v))(0);
                v -= c * Q.col(kk);
            }
            const double m_norm_sq = (v.transpose() * (M_sp * v))(0);
            if (m_norm_sq > 1.0e-20) {
                Q.col(q_cols) = v / std::sqrt(m_norm_sq);
                ++q_cols;
            }
        }
        Q.conservativeResize(Eigen::NoChange, q_cols);
        Eigen::MatrixXd K_proj_raw = Q.transpose() * (K_sp * Q);
        Eigen::MatrixXd M_proj_raw = Q.transpose() * (M_sp * Q);
        Eigen::MatrixXd K_proj = 0.5 * (K_proj_raw + K_proj_raw.transpose());
        Eigen::MatrixXd M_proj = 0.5 * (M_proj_raw + M_proj_raw.transpose());
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
            K_proj, M_proj);
        REQUIRE(ges.info() == Eigen::Success);
        const auto t3 = t_now();
        const double dt_multi = t_ms(t2, t3);

        std::cout << "  k=" << k << "  V=" << std::setw(5) << n_v
                  << "  3V=" << std::setw(5) << n_dof
                  << "  single=" << std::setw(8) << dt_single << " ms"
                  << "  shared=" << std::setw(8) << dt_multi << " ms"
                  << "  ratio=" << (dt_multi / dt_single) << '\n';
    }
    std::cout << std::defaultfloat;
}

TEST_CASE("Timing: single Spectra vs multi-seed fusion across k=2..5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 12. Production-cost comparison. The multi-seed fusion
    // heals the pentet bit-exactly, but does 3 Spectra runs + RR.
    // This probe times the EIGENSOLVE-ONLY portion (assembly is
    // amortized across both paths) for k = 2, 3, 4, 5.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    constexpr Eigen::Index nev_request = 24;

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    std::cout << "[timing single vs multi-seed]  nev=" << nev_request << '\n';
    std::cout << std::fixed << std::setprecision(3);

    for (int k : {2, 3, 4, 5}) {
        const auto mesh = cmsh::generate_icosphere(R, k);
        const Eigen::Index n_v = mesh.V.rows();
        const Eigen::Index n_dof = 3 * n_v;
        cs::LoopAssembler::Params p;
        p.use_stam = true;
        p.m_lump   = cs::MassLumping::None;
        const cs::LoopAssembler asm_(p);
        const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
        const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

        using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
        using OpM  = Spectra::SparseSymMatProd<double>;
        OpKM op_km(K_sp, M_sp);
        OpM  op_m(M_sp);
        const Eigen::Index ncv =
            std::min<Eigen::Index>(4 * nev_request + 1, K_sp.rows());

        auto t_now = []() {
            return std::chrono::high_resolution_clock::now();
        };
        auto t_ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // Single run (default seed).
        const auto t0 = t_now();
        Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
            solver_single(op_km, op_m, nev_request, ncv, kSigma);
        solver_single.init();
        solver_single.compute(Spectra::SortRule::LargestMagn,
                              /*maxit=*/1000, /*tol=*/1.0e-10);
        REQUIRE(solver_single.info() == Spectra::CompInfo::Successful);
        const auto t1 = t_now();
        const double dt_single = t_ms(t0, t1);

        // Multi-seed: 3 runs + RR fusion.
        const auto t2 = t_now();
        std::vector<Eigen::MatrixXd> V_runs;
        const uint32_t seeds[] = {0u, 1u, 0x3039u};
        for (uint32_t seed : seeds) {
            Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
                solver(op_km, op_m, nev_request, ncv, kSigma);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(-0.5, 0.5);
            Eigen::VectorXd init_vec(n_dof);
            for (Eigen::Index i = 0; i < n_dof; ++i) init_vec(i) = uni(rng);
            solver.init(init_vec.data());
            solver.compute(Spectra::SortRule::LargestMagn,
                           /*maxit=*/1000, /*tol=*/1.0e-10);
            REQUIRE(solver.info() == Spectra::CompInfo::Successful);
            V_runs.push_back(solver.eigenvectors());
        }
        Eigen::Index total_cols = 0;
        for (const auto& V : V_runs) total_cols += V.cols();
        Eigen::MatrixXd Combined(n_dof, total_cols);
        Eigen::Index off = 0;
        for (const auto& V : V_runs) {
            Combined.middleCols(off, V.cols()) = V;
            off += V.cols();
        }
        Eigen::MatrixXd Q(n_dof, Combined.cols());
        Eigen::Index q_cols = 0;
        for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
            Eigen::VectorXd v = Combined.col(j);
            for (Eigen::Index kk = 0; kk < q_cols; ++kk) {
                const double c = (Q.col(kk).transpose() * (M_sp * v))(0);
                v -= c * Q.col(kk);
            }
            const double m_norm_sq = (v.transpose() * (M_sp * v))(0);
            if (m_norm_sq > 1.0e-20) {
                Q.col(q_cols) = v / std::sqrt(m_norm_sq);
                ++q_cols;
            }
        }
        Q.conservativeResize(Eigen::NoChange, q_cols);
        Eigen::MatrixXd K_proj_raw = Q.transpose() * (K_sp * Q);
        Eigen::MatrixXd M_proj_raw = Q.transpose() * (M_sp * Q);
        Eigen::MatrixXd K_proj = 0.5 * (K_proj_raw + K_proj_raw.transpose());
        Eigen::MatrixXd M_proj = 0.5 * (M_proj_raw + M_proj_raw.transpose());
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
            K_proj, M_proj);
        REQUIRE(ges.info() == Eigen::Success);
        const auto t3 = t_now();
        const double dt_multi = t_ms(t2, t3);

        std::cout << "  k=" << k << "  V=" << std::setw(5) << n_v
                  << "  3V=" << std::setw(5) << n_dof
                  << "  single=" << std::setw(8) << dt_single << " ms"
                  << "  multi-seed=" << std::setw(8) << dt_multi << " ms"
                  << "  ratio=" << (dt_multi / dt_single) << '\n';
    }
    std::cout << std::defaultfloat;
}

TEST_CASE("Multi-seed Spectra runs + RR fusion heal the pentet?",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 11. The seed sweep (probe 10) showed different seeds
    // miss DIFFERENT directions of the n=2 invariant subspace —
    // seed 0 misses one direction (ghost at 5559 Hz); seed 0x3039
    // misses a different direction (ghost at 6529 Hz). If we
    // concatenate eigenvectors from MULTIPLE seed runs, the union
    // should span the full 5-D invariant subspace, and Rayleigh-
    // Ritz on the combined subspace should recover the clean
    // 5-fold pentet.
    //
    // Cost: K Spectra runs ≈ K seconds (cheap). Plus a dense RR on
    // a (K * nev)-dim subspace (instant). Total < 10 sec at k=5.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    const Eigen::Index n_dof = 3 * n_v;
    std::cout << "[multi-seed RR fusion, k=5 Stam, V=" << n_v << "]\n";

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using OpM  = Spectra::SparseSymMatProd<double>;
    OpKM op_km(K_sp, M_sp);
    OpM  op_m(M_sp);

    constexpr Eigen::Index nev = 24;
    constexpr Eigen::Index ncv = 100;

    // Collect eigenvectors from 3 seed runs.
    std::vector<Eigen::MatrixXd> V_runs;
    const uint32_t seeds[] = {0u, 1u, 0x3039u};
    for (uint32_t seed : seeds) {
        Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
            solver(op_km, op_m, nev, ncv, kSigma);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        Eigen::VectorXd init_vec(n_dof);
        for (Eigen::Index i = 0; i < n_dof; ++i) init_vec(i) = uni(rng);
        solver.init(init_vec.data());
        solver.compute(Spectra::SortRule::LargestMagn,
                       /*maxit=*/1000, /*tol=*/1.0e-10);
        REQUIRE(solver.info() == Spectra::CompInfo::Successful);
        V_runs.push_back(solver.eigenvectors());
    }

    Eigen::Index total_cols = 0;
    for (const auto& V : V_runs) total_cols += V.cols();
    Eigen::MatrixXd Combined(n_dof, total_cols);
    Eigen::Index off = 0;
    for (const auto& V : V_runs) {
        Combined.middleCols(off, V.cols()) = V;
        off += V.cols();
    }

    // M-orthonormalize via modified Gram-Schmidt; drop rank-deficient
    // columns (M-norm² < 1e-20 after deflation).
    Eigen::MatrixXd Q(n_dof, Combined.cols());
    Eigen::Index q_cols = 0;
    for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
        Eigen::VectorXd v = Combined.col(j);
        for (Eigen::Index k = 0; k < q_cols; ++k) {
            const double c = (Q.col(k).transpose() * (M_sp * v))(0);
            v -= c * Q.col(k);
        }
        const double m_norm_sq = (v.transpose() * (M_sp * v))(0);
        if (m_norm_sq > 1.0e-20) {
            Q.col(q_cols) = v / std::sqrt(m_norm_sq);
            ++q_cols;
        }
    }
    Q.conservativeResize(Eigen::NoChange, q_cols);
    std::cout << "  Combined subspace dim: " << q_cols
              << " (from " << total_cols << " candidate cols)\n";

    Eigen::MatrixXd K_proj = 0.5 * (Q.transpose() * (K_sp * Q)
                                   + (Q.transpose() * (K_sp * Q)).transpose());
    Eigen::MatrixXd M_proj = 0.5 * (Q.transpose() * (M_sp * Q)
                                   + (Q.transpose() * (M_sp * Q)).transpose());
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        K_proj, M_proj, Eigen::ComputeEigenvectors);
    REQUIRE(ges.info() == Eigen::Success);
    Eigen::VectorXd refined = ges.eigenvalues();
    std::cout << "  Multi-seed RR refined f (Hz), first 16:\n   ";
    for (Eigen::Index i = 0; i < 16 && i < refined.size(); ++i) {
        const double lam = std::max(0.0, refined(i));
        std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';
    // Pentet members live at indices 6..10 after the 6 RBMs.
    if (refined.size() >= 11) {
        Eigen::VectorXd p(5);
        for (int i = 0; i < 5; ++i) {
            const double lam = std::max(0.0, refined(6 + i));
            p(i) = std::sqrt(lam) / (2.0 * std::numbers::pi);
        }
        const double mean = p.mean();
        const double spread = (p.array() - mean).abs().maxCoeff();
        std::cout << "  pentet members: ";
        for (int i = 0; i < 5; ++i) std::cout << " " << p(i);
        std::cout << "\n  pentet_mean=" << mean
                  << "  rel_spread=" << (spread / mean) << '\n';
    }
}

TEST_CASE("Initial-residual seed sweep — does a different starting vector heal the pentet?",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 10. The nev sweep (probe 9) proved the pentet ghost is
    // bit-identical for all requested nev values — it's deterministic.
    // Spectra's default init() uses SimpleRandom<Scalar> with seed 0
    // (see SymEigsBase.h:311). The bug surfacing on icosahedrally
    // symmetric closed shells suggests starting-vector blindness:
    // the seed-0 random vector might have anomalously small
    // projection onto one of the 5 pentet invariant directions, so
    // Lanczos never captures it and produces the 5617 Hz ghost.
    //
    // Test: bypass the assembler path and run Spectra directly with
    // custom random initial residual vectors at several different
    // seeds. If different seeds give different ghost positions (or
    // sometimes no ghost), the bug is starting-vector blindness and
    // the production fix is a one-line init() change.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    const Eigen::Index n_dof = 3 * n_v;
    std::cout << "[seed sweep, k=5 Stam, V=" << n_v << "  3V=" << n_dof << "]\n";

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using OpM  = Spectra::SparseSymMatProd<double>;
    OpKM op_km(K_sp, M_sp);
    OpM  op_m(M_sp);

    constexpr Eigen::Index nev = 24;
    constexpr Eigen::Index ncv = 100;

    for (uint32_t seed : {0u, 1u, 42u, 12345u, 0xDEADBEEFu, 0xC1AD61u}) {
        Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
            solver(op_km, op_m, nev, ncv, kSigma);
        // Custom random initial residual at this seed.
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        Eigen::VectorXd init_vec(n_dof);
        for (Eigen::Index i = 0; i < n_dof; ++i) init_vec(i) = uni(rng);
        solver.init(init_vec.data());
        solver.compute(Spectra::SortRule::LargestMagn,
                       /*maxit=*/1000, /*tol=*/1.0e-10);
        if (solver.info() != Spectra::CompInfo::Successful) {
            std::cout << "  seed=" << seed << "  DID NOT CONVERGE\n";
            continue;
        }
        Eigen::VectorXd evals = solver.eigenvalues();
        std::sort(evals.data(), evals.data() + evals.size());
        // The first 6 are RBM residues (near zero, possibly negative).
        // The pentet starts at index 6.
        std::cout << "  seed=0x" << std::hex << seed << std::dec
                  << "  f (Hz) modes 6..12:";
        for (Eigen::Index i = 6; i <= 12 && i < evals.size(); ++i) {
            const double lam = std::max(0.0, evals(i));
            std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
        }
        if (evals.size() >= 11) {
            Eigen::VectorXd p(5);
            for (int i = 0; i < 5; ++i) {
                const double lam = std::max(0.0, evals(6 + i));
                p(i) = std::sqrt(lam) / (2.0 * std::numbers::pi);
            }
            const double mean = p.mean();
            const double spread = (p.array() - mean).abs().maxCoeff();
            std::cout << "  pentet_mean=" << mean
                      << "  rel_spread=" << (spread / mean);
        }
        std::cout << '\n';
    }
}

TEST_CASE("nev sweep — does the pentet ghost depend on Lanczos length?",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 9. Does the 5617 Hz ghost in the n=2 pentet at k=5 Stam
    // depend on the requested number of modes (n_modes_request)?
    // Two hypotheses to discriminate:
    //   (a) Boundary effect: the pentet sits near the edge of the
    //       requested nev range. Larger nev pushes it deeper into
    //       the well-converged region → ghost goes away.
    //   (b) Cumulative orthogonality drift: longer Lanczos = more
    //       drift in early-converged clusters → ghost gets WORSE
    //       with larger nev, BETTER with smaller nev.
    //
    // The pentet sits at eigenvalues #7-11 after the 6 RBMs. We
    // sweep n_modes_request from a tight (just covering the pentet)
    // to a generous range to map the dependence.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    std::cout << "[nev sweep, k=5 Stam, V=" << n_v << "]\n";

    for (std::size_t nm : {6u, 12u, 24u, 50u, 100u, 200u}) {
        const auto modes = cs::compute_shell_modes_loop(
            mesh.V, mesh.F, steel(), h_thk,
            nm, /*n_passes=*/1, /*use_stam=*/true);
        REQUIRE(modes.omegas.allFinite());
        // Pentet members are at indices 0..4 (after rigid filter).
        // Compute their spread.
        const Eigen::Index k_show = std::min<Eigen::Index>(6,
                                                            modes.omegas.size());
        std::cout << "  n_modes_request=" << nm
                  << "  nev (internal)=" << (nm + 16)
                  << "  first " << k_show << " physical f (Hz):";
        for (Eigen::Index i = 0; i < k_show; ++i) {
            std::cout << " " << modes.omegas(i) / (2.0 * std::numbers::pi);
        }
        // If the pentet has 5 members at indices 0..4, the ghost
        // is the one furthest from the mean of the other 4.
        if (modes.omegas.size() >= 5) {
            Eigen::VectorXd p(5);
            for (Eigen::Index i = 0; i < 5; ++i) {
                p(i) = modes.omegas(i) / (2.0 * std::numbers::pi);
            }
            const double mean = p.mean();
            const double spread = (p.array() - mean).abs().maxCoeff();
            std::cout << "  pentet_mean=" << mean
                      << "  spread=" << spread
                      << "  rel_spread=" << (spread / mean);
        }
        std::cout << '\n';
    }
}

TEST_CASE("Subspace iteration after Spectra at icosphere k=5 Stam",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 8. RR alone on Spectra's output can't recover the 5th
    // pentet member because Spectra's captured subspace is missing
    // that true direction (4 of 5 invariant directions captured).
    // Subspace iteration: apply the shift-invert operator
    // W = (K - σM)⁻¹ M V once to push the captured vectors deeper
    // into the invariant subspace. The fixed-point of subspace
    // iteration IS the invariant subspace, so one application
    // should be enough to recover any direction the Krylov iteration
    // missed by orthogonality drift — as long as that direction has
    // non-zero projection onto the SI operator's image of V.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    constexpr double kSigma = -1.0;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    std::cout << "[Subspace-iteration probe, k=5 Stam, V=" << n_v << "]\n";

    constexpr std::size_t n_modes_request = 50;
    const auto modes = cs::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h_thk,
        n_modes_request, /*n_passes=*/1, /*use_stam=*/true);
    REQUIRE(modes.omegas.allFinite());

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;
    cs::LoopAssembler::Params p;
    p.use_stam = true;
    p.m_lump   = cs::MassLumping::None;
    const cs::LoopAssembler asm_(p);
    const auto K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
    const auto M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

    const Eigen::MatrixXd& V_modes = modes.shapes;

    // Factor (K - σM). σ = -1, K PSD, M SPD ⇒ (K - σM) is SPD.
    Eigen::SparseMatrix<double> K_shifted = K_sp - kSigma * M_sp;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(K_shifted);
    if (lu.info() != Eigen::Success) {
        std::cout << "  SparseLU factor failed.\n";
        return;
    }

    // W = (K - σM)⁻¹ M V — subspace iteration step.
    Eigen::MatrixXd MV = M_sp * V_modes;
    Eigen::MatrixXd W = lu.solve(MV);

    // Concatenate the original Spectra subspace with the SI-updated
    // one, so the projected RR has both populations: 2 * n_modes_request
    // = 100 candidate directions.
    Eigen::MatrixXd Combined(V_modes.rows(),
                              V_modes.cols() + W.cols());
    Combined.leftCols(V_modes.cols())   = V_modes;
    Combined.rightCols(W.cols())         = W;

    // M-orthonormalize Combined (drop linearly dependent columns).
    // Use modified Gram-Schmidt in the M-inner-product. A column is
    // dropped if its M-norm-squared falls below 1e-20 after
    // orthogonalization against earlier columns.
    Eigen::MatrixXd Q(Combined.rows(), 0);
    Q.resize(Combined.rows(), Combined.cols());
    Eigen::Index q_cols = 0;
    for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
        Eigen::VectorXd v = Combined.col(j);
        for (Eigen::Index k = 0; k < q_cols; ++k) {
            const double c = (Q.col(k).transpose() * (M_sp * v))(0);
            v -= c * Q.col(k);
        }
        const double m_norm_sq = (v.transpose() * (M_sp * v))(0);
        if (m_norm_sq > 1.0e-20) {
            Q.col(q_cols) = v / std::sqrt(m_norm_sq);
            ++q_cols;
        }
    }
    Q.conservativeResize(Eigen::NoChange, q_cols);
    std::cout << "  Subspace dim after M-orthonormalization: "
              << q_cols << " (started from "
              << Combined.cols() << ")\n";

    Eigen::MatrixXd K_proj = Q.transpose() * (K_sp * Q);
    Eigen::MatrixXd M_proj = Q.transpose() * (M_sp * Q);  // should be ≈ I
    K_proj = 0.5 * (K_proj + K_proj.transpose());
    M_proj = 0.5 * (M_proj + M_proj.transpose());

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        K_proj, M_proj, Eigen::ComputeEigenvectors);
    REQUIRE(ges.info() == Eigen::Success);
    const Eigen::VectorXd refined = ges.eigenvalues();
    std::cout << "  Subspace-iteration + RR refined f (Hz), first 16:\n   ";
    for (Eigen::Index i = 0; i < 16 && i < refined.size(); ++i) {
        const double lam = std::max(0.0, refined(i));
        std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
    }
    std::cout << '\n';
}

TEST_CASE("Stam SymGEigsSolver<Cholesky> at icosphere k=4 and k=5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 6. The other non-deprecated Spectra solver mode for the
    // generalized eigenproblem K φ = λ M φ. Cholesky mode factors
    // M = L L^T and runs Lanczos on the standard symmetric problem
    // L^-1 K L^-T y = λ y. Different shift handling than the failing
    // ShiftInvert path — worth measuring before concluding "all
    // Spectra options fail".
    //
    // Caveat: Lanczos converges to EXTREMAL eigenvalues. Without
    // shift-invert, the natural convergence is toward LARGEST
    // eigenvalues. SortRule::SmallestAlge requests the bottom of the
    // spectrum, which converges more slowly but, crucially, with a
    // DIFFERENT orthogonality profile on degenerate clusters than
    // shift-invert. Empirically (this probe) Cholesky mode gives a
    // clean 5-fold pentet on the k=4 Stam fixture where ShiftInvert
    // returns 4+1. Tested at k=4 AND k=5 below — k=5 is where the
    // ShiftInvert bug was dramatic (5% spurious split).
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;

    const auto mat = steel();
    const auto sm  = cs::shell_material_from_isotropic(mat, h_thk);
    const double rho_h = mat.density * h_thk;

    for (int k : {4, 5}) {
        const auto mesh = cmsh::generate_icosphere(R, k);
        std::cout << "[Spectra Cholesky probe, k=" << k << " Stam, V="
                  << mesh.V.rows() << "]\n";

        cs::LoopAssembler::Params p;
        p.use_stam = true;
        p.m_lump   = cs::MassLumping::None;
        const cs::LoopAssembler asm_(p);
        Eigen::SparseMatrix<double> K_sp = asm_.assemble_K(mesh.V, mesh.F, sm);
        Eigen::SparseMatrix<double> M_sp = asm_.assemble_M(mesh.V, mesh.F, rho_h);

        using OpK = Spectra::SparseSymMatProd<double>;
        using OpM = Spectra::SparseCholesky<double>;
        OpK op_k(K_sp);
        OpM op_m(M_sp);
        if (op_m.info() != Spectra::CompInfo::Successful) {
            std::cout << "  SparseCholesky of M failed.\n";
            continue;
        }

        constexpr int nev = 12;
        constexpr int ncv = 60;
        Spectra::SymGEigsSolver<OpK, OpM, Spectra::GEigsMode::Cholesky>
            geigs(op_k, op_m, nev, ncv);
        geigs.init();
        const int nconv = geigs.compute(Spectra::SortRule::SmallestAlge,
                                         /*maxit=*/5000, /*tol=*/1e-10);
        std::cout << "  Cholesky info = " << static_cast<int>(geigs.info())
                  << " (Successful=" << static_cast<int>(Spectra::CompInfo::Successful)
                  << ")  nconv=" << nconv << '\n';
        if (geigs.info() != Spectra::CompInfo::Successful) {
            std::cout << "  Cholesky mode did NOT converge at k=" << k << ".\n";
            continue;
        }
        Eigen::VectorXd evals = geigs.eigenvalues();
        // Spectra returns them in convergence order; sort ascending
        // by algebraic value for human-readable output.
        std::sort(evals.data(), evals.data() + evals.size());
        std::cout << "  smallest 12 eigenvalues (sorted ascending):\n   ";
        for (int i = 0; i < evals.size(); ++i) std::cout << " " << evals(i);
        std::cout << "\n  Corresponding f (Hz):\n   ";
        for (int i = 0; i < evals.size(); ++i) {
            const double lam = std::max(0.0, evals(i));
            std::cout << " " << std::sqrt(lam) / (2.0 * std::numbers::pi);
        }
        std::cout << '\n';
    }
    std::cout << "  Reference (dense @ k=4): 6 RBMs near 0, then 5×5911.68 Hz, then 7002.33 Hz\n"
              << "  Reference (Wilkinson):   n=2 pentet at 5903.22 Hz\n";
}

TEST_CASE("Stam spurious-mode localisation at icosphere k=5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe 1 of the Stam bug investigation. The Stam path produces
    // a spurious low mode at ~5617 Hz on icosphere k=5 that L.3.4
    // doesn't have. If the mode shape concentrates at the 12
    // valence-5 vertices (indices 0..11 by generate_icosphere's
    // construction), the bug is local to the Stam patch kernel in
    // loop_stam.cpp. If it's spread globally, the bug is in the
    // assembler / global eigenproblem.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 5);
    const Eigen::Index n_v = mesh.V.rows();
    std::cout << "[Stam spurious-mode localisation — k=5  V="
              << n_v << "]\n";

    const auto modes = cs::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h,
        /*n_modes=*/4, /*n_passes=*/1, /*use_stam=*/true);
    REQUIRE(modes.omegas.allFinite());

    // For each mode, compute per-vertex amplitude (Euclidean norm of
    // the 3-vector displacement), then compare valence-5 (indices
    // 0..11) to valence-6 (indices 12..) amplitudes.
    constexpr Eigen::Index N_v5 = 12;
    for (Eigen::Index k = 0; k < modes.omegas.size(); ++k) {
        const Eigen::VectorXd col = modes.shapes.col(k);
        Eigen::VectorXd amp(n_v);
        for (Eigen::Index v = 0; v < n_v; ++v) {
            amp(v) = std::sqrt(col(3*v)*col(3*v)
                             + col(3*v+1)*col(3*v+1)
                             + col(3*v+2)*col(3*v+2));
        }
        double v5_mean = 0.0, v5_max = 0.0;
        for (Eigen::Index v = 0; v < N_v5; ++v) {
            v5_mean += amp(v);
            v5_max  = std::max(v5_max, amp(v));
        }
        v5_mean /= static_cast<double>(N_v5);

        double v6_mean = 0.0, v6_max = 0.0;
        for (Eigen::Index v = N_v5; v < n_v; ++v) {
            v6_mean += amp(v);
            v6_max  = std::max(v6_max, amp(v));
        }
        v6_mean /= static_cast<double>(n_v - N_v5);

        std::cout << "  mode " << k << "  f="
                  << modes.omegas(k) / (2.0 * std::numbers::pi)
                  << " Hz  |  v5(mean,max)=(" << v5_mean << ","
                  << v5_max << ")  v6(mean,max)=(" << v6_mean
                  << "," << v6_max << ")  ratio v5_max/v6_max="
                  << (v6_max > 0 ? v5_max / v6_max : 0.0) << '\n';
    }
}

TEST_CASE("Stam-vs-L.3.4 spectrum dump at icosphere k=4..5",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Diagnostic — Stam exact-eval drifts hard at k=5 (rel_err
    // jumped from 0.064% at k=4 to 0.937% at k=5 while L.3.4 stayed
    // within 0.05%). Dump the first 12 omegas to identify whether
    // a spurious low mode is appearing in the Stam path.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const double f_w = wilkinson_n2_hz();
    std::cout << "[Wilkinson n=2 = " << f_w << " Hz]\n";

    for (int k : {4, 5}) {
        const auto mesh = cmsh::generate_icosphere(R, k);
        std::cout << "  k=" << k << "  V=" << mesh.V.rows() << '\n';
        for (bool use_stam : {false, true}) {
            const auto modes = cs::compute_shell_modes_loop(
                mesh.V, mesh.F, steel(), h,
                /*n_modes=*/12, /*n_passes=*/1, use_stam);
            REQUIRE(modes.omegas.allFinite());
            std::cout << "    " << (use_stam ? "Stam " : "L.3.4")
                      << " first 12 omegas/2π (Hz):\n     ";
            for (Eigen::Index i = 0; i < modes.omegas.size(); ++i) {
                std::cout << " " << modes.omegas(i) / (2.0 * std::numbers::pi);
            }
            std::cout << '\n';
        }
    }
}

TEST_CASE("Projection closes the icosphere FEM-vs-Wilkinson gap",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    const double f_w = wilkinson_n2_hz();
    auto rel_err = [&](double f) { return std::abs(f - f_w) / f_w; };

    struct Row {
        int k;
        bool project;
        int n_passes;
        bool use_stam;
        const char* tag;
    };
    const Row rows[] = {
        {2, false, 1, false, "k=2 raw   L.3.4 p1 "},
        {2, false, 2, false, "k=2 raw   L.3.4 p2 "},
        {2, true,  1, false, "k=2 proj  L.3.4 p1 "},
        {2, true,  2, false, "k=2 proj  L.3.4 p2 "},
        {2, true,  1, true,  "k=2 proj  Stam  p1 "},
        {3, false, 1, false, "k=3 raw   L.3.4 p1 "},
        {3, false, 2, false, "k=3 raw   L.3.4 p2 "},
        {3, true,  1, false, "k=3 proj  L.3.4 p1 "},
        {3, true,  1, true,  "k=3 proj  Stam  p1 "},
    };

    std::cout << "[icosphere FEM vs Wilkinson n=2 = " << f_w << " Hz]\n";
    for (const auto& r : rows) {
        const double f = n2_mean_hz_for_icosphere(
            r.k, r.project, r.n_passes, r.use_stam);
        std::cout << "  " << r.tag
                  << "  f_n2_mean = " << f << " Hz"
                  << "  rel_err = " << rel_err(f) << '\n';
    }
}

TEST_CASE("Projection preserves NaN-free spectrum across k=2..3",
          "[shell][loop][sphere_projected][regression]")
{
    // Cheap regression: just verify the projected-mesh path produces
    // finite spectra at low k.
    const double f = n2_mean_hz_for_icosphere(
        /*k=*/2, /*project=*/true, /*n_passes=*/1, /*use_stam=*/false);
    REQUIRE(std::isfinite(f));
    REQUIRE(f > 0.0);
}

// Projected-k=3 case demoted to [.experiment] sweep above — it
// tests the same regime (2562 V on sphere) as raw k=4 below but
// via a different vertex distribution (Loop-smooth-then-project
// instead of midpoint-subdivision-then-project). Both achieve
// sub-percent rel_err. The raw-k=4 test below is the canonical
// closed-shell validation.

TEST_CASE("Mass-lumping effect on icosphere(k=2,3) Loop modes vs Wilkinson",
          "[shell][loop][sphere_projected][experiment][.experiment]")
{
    // Probe: quantify how much of the ~35 % Loop-vs-Wilkinson softness
    // at k=2 (under the @b former @ref MassLumping::RowSum default,
    // pre-2026-05-17 late) was due to lumping vs the genuine geometric
    // Loop-limit-surface effect. Findings:
    //
    //   icosphere k=2  RowSum:        rel_err = 35.0 %
    //   icosphere k=2  None:          rel_err =  2.3 %
    //   icosphere k=3  RowSum:        rel_err = 18.2 %
    //   icosphere k=3  None:          rel_err =  0.3 %
    //
    // Lumping accounts for nearly all of the historical 35 % gap; the
    // residual ~2 % at k=2 is the genuine geometric error plus the
    // independent coarse-mesh approximation error of the FEM. Default
    // is now @ref MassLumping::None; this probe stays as a permanent
    // diagnostic of the trade.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto       mat = steel();
    const auto       sm  = cs::shell_material_from_isotropic(mat, h);
    const double f_w = wilkinson_n2_hz();
    std::cout << "[lumping_probe Wilkinson n=2 = " << f_w << " Hz]\n";

    auto run = [&](int k, cs::MassLumping lump) {
        const auto mesh = cmsh::generate_icosphere(R, k);
        cs::LoopAssembler::Params p;
        p.m_lump = lump;
        cs::LoopAssembler asm_(p);
        const auto modes = cs::compute_shell_modes(
            mesh.V, mesh.F, mat, sm, h, /*n_modes=*/6, asm_);
        double sum = 0.0;
        for (Eigen::Index i = 0; i < 5; ++i) sum += modes.omegas(i);
        return (sum / 5.0) / (2.0 * std::numbers::pi);
    };

    for (int k : {2, 3}) {
        const double f_lumped     = run(k, cs::MassLumping::RowSum);
        const double f_consistent = run(k, cs::MassLumping::None);
        const double rel_l = std::abs(f_lumped     - f_w) / f_w;
        const double rel_c = std::abs(f_consistent - f_w) / f_w;
        std::cout << "  k=" << k
                  << "  lumped     f=" << f_lumped     << "  rel_err=" << rel_l << '\n'
                  << "        consistent f=" << f_consistent << "  rel_err=" << rel_c
                  << "  (Δ from lumping = " << (f_consistent - f_lumped)
                  << " Hz = " << (f_consistent - f_lumped) / f_lumped * 100.0
                  << "%)\n";
    }
}

TEST_CASE("Stam K on icosphere k=4: clean 5-fold n=2 pentet under multi-seed solver",
          "[shell][loop][sphere_projected][analytical][validation]")
{
    // Regression guard for the multi-seed Spectra fix to the degenerate
    // n=2 pentet cluster bug (memory: chladni-spectra-degenerate-cluster).
    // Before the multi-seed eigensolve landed,
    // solve_modal_eigenproblem_with_rigid_filter returned a 4+1 split on
    // icosphere k=4 Stam — 4 modes at 5911.68 Hz + 1 ghost at 5850.46 Hz.
    // The dense reference confirms the true spectrum is a clean 5-fold
    // pentet at 5911.68 Hz; this test asserts the solver actually
    // returns that.
    //
    // Threshold rationale: rel_spread < 1e-10 catches any return of the
    // ghost (5850 vs 5912 = ~1% spread = 1e-2). Multi-seed empirically
    // delivers ~3e-15 on this fixture, leaving a wide safety margin.
    constexpr double R = 0.10;
    constexpr double h_thk = 1.0e-3;
    const auto mesh = cmsh::generate_icosphere(R, 4);
    constexpr std::size_t n_modes_request = 12;
    const auto modes = cs::compute_shell_modes_loop(
        mesh.V, mesh.F, steel(), h_thk,
        n_modes_request, /*n_passes=*/1, /*use_stam=*/true);
    REQUIRE(modes.omegas.size() == static_cast<Eigen::Index>(n_modes_request));
    REQUIRE(modes.omegas.allFinite());

    // n=2 pentet members are at indices 0..4 after the rigid filter.
    Eigen::VectorXd pentet(5);
    for (int i = 0; i < 5; ++i) {
        pentet(i) = modes.omegas(i) / (2.0 * std::numbers::pi);
    }
    const double mean = pentet.mean();
    const double spread = (pentet.array() - mean).abs().maxCoeff();
    const double rel_spread = spread / mean;
    INFO("pentet f (Hz) = " << pentet.transpose()
         << "  mean=" << mean << "  rel_spread=" << rel_spread);
    REQUIRE(rel_spread < 1.0e-10);

    // And the pentet mean matches the dense reference (5911.68 Hz)
    // within FEM convergence error (Stam at k=4 vs Wilkinson is ~0.06%).
    constexpr double f_dense_ref = 5911.68;
    REQUIRE(std::abs(mean - f_dense_ref) / f_dense_ref < 1.0e-3);
}

TEST_CASE("Raw icosphere(k=4) FEM ≈ Wilkinson within 0.5%",
          "[shell][loop][sphere_projected][analytical][validation]")
{
    // The clean version of the closed-shell FEM validation: every
    // vertex of icosphere(k=4) sits exactly on the sphere of radius
    // R by construction (no projection trick mid-flight). At
    // V=2562, the rel_err drops to ~0.2% on L.3.4 — comparable in
    // tightness to the cylinder/plate analytic fixtures. Pins the
    // full Cirak-Ortiz Loop pipeline end-to-end on a closed shell.
    //
    // Stam path achieves similar rel_err (~0.3%) but is NOT
    // robustly better here — see the [.experiment] sweep above for
    // the data. Stam ≈ L.3.4 holds on sphere meshes (S.8 finding
    // confirmed).
    const double f_w = wilkinson_n2_hz();
    const double f = n2_mean_hz_for_icosphere(
        /*k=*/4, /*project=*/false, /*n_passes=*/1, /*use_stam=*/false);
    const double rel_err = std::abs(f - f_w) / f_w;
    INFO("Wilkinson = " << f_w << " Hz, FEM = " << f << " Hz, "
         "rel_err = " << rel_err);
    REQUIRE(rel_err < 5.0e-3);  // 0.5% — margin over the measured 0.2%
}
