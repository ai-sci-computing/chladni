/**
 * @file test_rigid_filter_distribution.cpp
 * @brief R8 measure-first probe: dump the rigid-body mass-projection
 *        rigid_proj_sq distribution across the leading spectrum of several
 *        closed / nearly-closed shells.
 *
 * The production rigid filter
 * (chladni::shell::solve_modal_eigenproblem_with_rigid_filter) classifies a
 * mode as rigid iff its squared M-norm projection onto the 6-dim rigid
 * subspace exceeds a HARD threshold of 0.5. The review (R8) worried this
 * could drop a 60 %-rigid physical mode or keep a 49 %-contaminated one. The
 * code comment claims the projection is sharply bimodal (physical ~1e-6,
 * rigid ~1.0) so 0.5 is unambiguous. This probe measures the real
 * distribution to decide whether an adaptive gap rule is needed or the hard
 * threshold is genuinely safe.
 *
 * RESULT (2026-06-09): rigid_proj_sq is machine-perfectly BIMODAL on every
 * mesh measured — closed icosphere k=2/k=3 and the free-free cylinder (the
 * case that historically broke the frequency-gap heuristic). The 6 rigid
 * modes read exactly 1.0; every physical mode reads <= ~1e-22. The margin
 * around 0.5 is 1.0 (≈22+ orders of magnitude). This is structural, not
 * luck: K's exact null space IS the 6-dim rigid subspace, so for the
 * symmetric (K, M) pencil every physical eigenvector (a distinct eigenvalue)
 * is M-ORTHOGONAL to the rigid kernel and has identically-zero projection.
 * The review's feared 60 %-rigid / 49 %-contaminated mode cannot occur. The
 * hard 0.5 threshold is therefore kept; an adaptive gap rule would be a
 * no-op. This test is the standing regression guard for that invariant.
 * Set RIGID_DIAG=1 to print the per-mode projections.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

chladni::IsotropicMaterial steel()
{
    return chladni::IsotropicMaterial{
        .youngs_modulus = 200.0e9, .poisson_ratio = 0.30, .density = 7850.0};
}

/// rigid_proj_sq for every eigenvector, computed exactly as the production
/// filter does (c = (M V_R)^T phi, rigid_proj_sq = c^T G^{-1} c with
/// G = V_R^T M V_R), but on a DENSE solve of the WHOLE spectrum so the
/// would-be-rejected rigid cluster is visible too.
std::vector<double> rigid_projections(const chladni::mesh::TriMesh& mesh,
                                      double h,
                                      int n_report)
{
    const chladni::shell::LoopAssembler assembler{};
    const auto sm = chladni::shell::shell_material_from_isotropic(steel(), h);
    const Eigen::MatrixXd K = Eigen::MatrixXd(assembler.assemble_K(mesh.V, mesh.F, sm));
    const Eigen::MatrixXd M = Eigen::MatrixXd(
        assembler.assemble_M(mesh.V, mesh.F, steel().density * h));

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
        K, M, Eigen::ComputeEigenvectors);
    REQUIRE(ges.info() == Eigen::Success);
    const Eigen::MatrixXd evecs = ges.eigenvectors();

    Eigen::MatrixXd V_rigid = Eigen::MatrixXd::Zero(3 * mesh.V.rows(), 6);
    for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
        const double x = mesh.V(i, 0), y = mesh.V(i, 1), z = mesh.V(i, 2);
        V_rigid(3 * i + 0, 0) = 1.0;
        V_rigid(3 * i + 1, 1) = 1.0;
        V_rigid(3 * i + 2, 2) = 1.0;
        V_rigid(3 * i + 1, 3) = -z; V_rigid(3 * i + 2, 3) = y;
        V_rigid(3 * i + 0, 4) =  z; V_rigid(3 * i + 2, 4) = -x;
        V_rigid(3 * i + 0, 5) = -y; V_rigid(3 * i + 1, 5) = x;
    }
    const Eigen::MatrixXd MV = M * V_rigid;
    const Eigen::Matrix<double, 6, 6> G = V_rigid.transpose() * MV;
    const Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(G);

    std::vector<double> proj(static_cast<std::size_t>(evecs.cols()));
    for (Eigen::Index k = 0; k < evecs.cols(); ++k) {
        const Eigen::Matrix<double, 6, 1> c = MV.transpose() * evecs.col(k);
        proj[static_cast<std::size_t>(k)] = c.dot(chol.solve(c));
    }
    if (std::getenv("RIGID_DIAG")) {
        for (int k = 0; k < n_report && k < static_cast<int>(proj.size()); ++k) {
            std::printf("    mode %2d  lambda=%.4e  rigid_proj_sq=%.6e\n",
                        k, ges.eigenvalues()(k),
                        proj[static_cast<std::size_t>(k)]);
        }
    }
    return proj;
}

/// The smallest gap, in the rigid-projection axis, separating the cluster
/// below 0.5 from the cluster above it. A large gap means the hard 0.5
/// threshold has wide margin; a small gap (a mode landing near 0.5) is the
/// borderline case the review feared.
double margin_around_half(const std::vector<double>& proj)
{
    double max_below = 0.0, min_above = 1.0;
    for (double p : proj) {
        if (p <= 0.5) max_below = std::max(max_below, p);
        else          min_above = std::min(min_above, p);
    }
    return min_above - max_below;
}

}  // namespace

TEST_CASE("rigid filter: rigid_proj_sq stays bimodal on closed shells (R8 guard)",
          "[shell][rigid_dist]")
{
    constexpr double h = 1.0e-3;

    struct Case { const char* name; chladni::mesh::TriMesh mesh; };
    std::vector<Case> cases;
    cases.push_back({"icosphere k=2",
                     chladni::mesh::generate_icosphere(0.10, 2)});
    cases.push_back({"icosphere k=3",
                     chladni::mesh::generate_icosphere(0.10, 3)});
    // Free-free cylinder: the comment in solve_modal_eigenproblem_with_
    // rigid_filter cites this as the case that broke the old frequency-gap
    // heuristic — a spurious rigid-rotation residue that lands mid-spectrum.
    // Confirm the mass-projection still classifies it unambiguously.
    cases.push_back({"free-free cylinder 16x8",
                     chladni::mesh::generate_cylinder(0.10, 0.30, 16, 8)});

    for (auto& c : cases) {
        std::printf("[rigid_dist] %s (%lld V):\n",
                    c.name, static_cast<long long>(c.mesh.V.rows()));
        const auto proj = rigid_projections(c.mesh, h, 20);
        const double margin = margin_around_half(proj);
        if (std::getenv("RIGID_DIAG")) {
            std::printf("[rigid_dist] %s: margin around 0.5 = %.4e\n",
                        c.name, margin);
        }
        // The invariant the hard 0.5 threshold relies on: a wide bimodal
        // separation. No mode lands in the ambiguous band, and the gap
        // straddling 0.5 is enormous (physical projections are ~machine
        // zero, rigid ~1.0). If this ever fails, the spectral structure has
        // changed and the 0.5 threshold — or this whole analysis — needs a
        // fresh look (that would be the borderline fixture the review feared).
        INFO(c.name << ": margin around 0.5 = " << margin);
        CHECK(margin > 0.9);
        for (double p : proj) {
            const bool ambiguous = (p > 0.1 && p < 0.9);
            INFO("rigid_proj_sq = " << p);
            CHECK_FALSE(ambiguous);
        }
    }
}
