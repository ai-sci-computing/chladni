#pragma once

/**
 * @file lme_test_helpers.hpp
 * @brief Shared LME eigensolve helper for the LME-vs-analytic tests.
 *
 * Builds @c K and @c M with @ref chladni::shell::LMEAssembler, extracts
 * the @c z-component sub-block (the LMEAssembler ships three identical
 * scalar tiles; for the Kirchhoff plate the physical spectrum lives on
 * the @c z block), optionally pins boundary @c z-DOFs, and runs a
 * sparse Spectra shift-invert solve on the @f$ n_v \times n_v @f$
 * reduced pencil. Returns the lowest @p n_compute eigenvalues sorted
 * ascending.
 *
 * Why sparse Spectra and not Eigen's @c GeneralizedSelfAdjointEigenSolver?
 * The disk fixtures run at ~250 vertices baseline and ~1400 vertices in
 * convergence sweeps. Dense @f$ n_v^3 @f$ is wasteful at these sizes
 * (and 30 s+ at 64×16). Spectra's shift-invert finds the lowest cluster
 * directly with one sparse factorisation, matching the path used by
 * @ref chladni::shell::solve_modal_eigenproblem_with_rigid_filter for
 * the Loop pipeline.
 */

#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace chladni::tests::lme {

/// Lowest few eigenvalues of the z-block LME generalized eigenproblem
/// on a flat plate, sorted ascending. Eigenvectors are not returned.
struct ZBlockModes {
    Eigen::VectorXd eigenvalues;  ///< ascending, length n_compute
    Eigen::Index    n_v_total;    ///< vertices in the input mesh
    Eigen::Index    n_z_free;     ///< unpinned z-DOFs solved on
};

/**
 * @brief Solve the z-block generalized eigenproblem on a flat-plate
 *        LME assembly.
 *
 * @param V                       Vertex positions (flat plate; z must
 *                                be constant — checked by the assembler).
 * @param F                       Triangulation; defines the integration
 *                                domain.
 * @param sm                      Bending stiffness coefficients.
 * @param surface_density         @f$ \rho h @f$.
 * @param z_bdry_pinned           For each vertex, @c true to pin its
 *                                @c z-DOF to 0 (Dirichlet @f$ w = 0 @f$).
 *                                Pass @c std::vector<bool>(n_v, false)
 *                                for free-edge BC. Must be length @c V.rows().
 * @param n_compute               Number of lowest eigenvalues to return.
 * @param params                  LMEAssembler params (γ, r_cut_mult,
 *                                Newton tolerance, …). Default-constructed
 *                                gives Millán 2011 defaults.
 * @param rigid_slack             Extra Krylov room above @c n_compute
 *                                so the leading rigid cluster fits.
 *                                Default 16 mirrors @ref
 *                                solve_modal_eigenproblem_with_rigid_filter.
 *
 * @throws std::runtime_error on Spectra non-convergence.
 */
[[nodiscard]] inline ZBlockModes solve_lme_z_modes(
    const Eigen::MatrixXd&                       V,
    const Eigen::MatrixXi&                       F,
    const chladni::shell::ShellMaterial&         sm,
    double                                       surface_density,
    const std::vector<bool>&                     z_bdry_pinned,
    std::size_t                                  n_compute,
    chladni::shell::LMEAssembler::Params         params = {},
    Eigen::Index                                 rigid_slack = 16)
{
    using SparseD = Eigen::SparseMatrix<double>;
    const Eigen::Index n_v = V.rows();

    chladni::shell::LMEAssembler lme(params);
    const auto K_full = lme.assemble_K(V, F, sm);
    const auto M_full = lme.assemble_M(V, F, surface_density);

    // Z-block selection + Dirichlet pinning: pick row 3*v+2 for every
    // unpinned vertex v.
    std::vector<Eigen::Index> z_free;
    z_free.reserve(static_cast<std::size_t>(n_v));
    for (Eigen::Index v = 0; v < n_v; ++v) {
        if (!z_bdry_pinned[static_cast<std::size_t>(v)]) {
            z_free.push_back(3 * v + 2);
        }
    }
    const Eigen::Index n_free =
        static_cast<Eigen::Index>(z_free.size());

    SparseD P(3 * n_v, n_free);
    P.reserve(Eigen::VectorXi::Constant(n_free, 1));
    for (Eigen::Index k = 0; k < n_free; ++k) {
        P.insert(z_free[static_cast<std::size_t>(k)], k) = 1.0;
    }
    SparseD K_red = P.transpose() * K_full * P;
    SparseD M_red = P.transpose() * M_full * P;
    K_red.makeCompressed();
    M_red.makeCompressed();

    // ----- Spectra shift-invert solve -------------------------------
    using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using OpM  = Spectra::SparseSymMatProd<double>;
    OpKM op_km(K_red, M_red);
    OpM  op_m(M_red);

    const Eigen::Index nev =
        static_cast<Eigen::Index>(n_compute) + rigid_slack;
    if (nev > n_free) {
        throw std::runtime_error(
            "solve_lme_z_modes: n_compute + rigid_slack exceeds "
            "number of free DOFs");
    }
    const Eigen::Index ncv =
        std::min<Eigen::Index>(
            std::max<Eigen::Index>(4 * nev + 1, 60), n_free);
    constexpr double kSigma = -1.0;

    Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
        solver(op_km, op_m, nev, ncv, kSigma);
    solver.init();
    constexpr int    kMaxIters  = 1000;
    constexpr double kSolverTol = 1.0e-10;
    solver.compute(Spectra::SortRule::LargestMagn, kMaxIters, kSolverTol);
    if (solver.info() != Spectra::CompInfo::Successful) {
        throw std::runtime_error(
            "solve_lme_z_modes: Spectra shift-invert did not converge");
    }

    Eigen::VectorXd evals_raw = solver.eigenvalues();
    std::vector<double> evals_sorted(evals_raw.data(),
                                      evals_raw.data() + evals_raw.size());
    std::sort(evals_sorted.begin(), evals_sorted.end());

    ZBlockModes out;
    out.n_v_total = n_v;
    out.n_z_free  = n_free;
    out.eigenvalues.resize(static_cast<Eigen::Index>(n_compute));
    for (std::size_t k = 0; k < n_compute; ++k) {
        out.eigenvalues(static_cast<Eigen::Index>(k)) = evals_sorted[k];
    }
    return out;
}

}  // namespace chladni::tests::lme
