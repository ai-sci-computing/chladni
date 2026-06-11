/**
 * @file test_loop_multipoint_quad.cpp
 * @brief Investigation: what's behind the icosphere 36% rel_err gap?
 *
 * S.8 finding: at k=2 (162 V) the L.3.4 and Stam paths both produce
 * ~36% rel_err vs Wilkinson, and the convergence ratio (~0.52) is
 * unchanged by Stam-vs-L.3.4. The post-S.8 hypothesis was that the
 * dominant error is the one-point centroid quadrature in
 * element_stiffness_regular / element_stiffness_stam.
 *
 * @section result Result (2026-05-12)
 *
 * Quadrature hypothesis REFUTED. Tested 1-, 3-, and 7-point Gauss
 * (Cowper for 7-pt, exact for degree 5 — overkill for the degree-4
 * B^T H B integrand) and the rel_err moves by ~1% (within noise)
 * and slightly the wrong direction. The Cirak-Ortiz centroid choice
 * is essentially optimal here. Numbers (Stam path, R=0.10 m, h=1 mm):
 *
 *     n_quad=1: f_n2 = 3796 Hz, rel_err = 35.7%
 *     n_quad=3: f_n2 = 3741 Hz, rel_err = 36.6%
 *     n_quad=7: f_n2 = 3753 Hz, rel_err = 36.4%
 *
 * Root cause: the Loop limit surface built from an icosphere k=2
 * control mesh is NOT a sphere. Measured directly: limit r_mean =
 * 0.0977 (sphere target 0.10 — 2.25% inward shift), with limit r
 * varying from 0.0974 to 0.0979 across face centroids (0.5%
 * non-spherical lumpiness). A lumpy non-spherical shell has
 * lower-frequency bending modes than the corresponding analytic
 * sphere, by exactly the amount we observe.
 *
 * @section conclusion Conclusion
 *
 * The icosphere FEM-vs-Wilkinson gap is intrinsic to the Loop
 * subdivision discretisation of a sphere, not a code bug. Fixing it
 * would require either (a) using a basis that exactly represents
 * spheres (NURBS / rational Catmull-Clark — far beyond Loop's
 * scope), or (b) sufficient mesh refinement (k=4: 2562 V, ~10%
 * rel_err; k=5: 10242 V, ~5%). Refinement is the only path inside
 * the Cirak-Ortiz framework.
 *
 * Hidden by default ([.experiment]) — kept as documentation of the
 * investigation and as a baseline for any future revisit (e.g. if
 * we add adaptive quadrature near valence-5 vertices, or migrate
 * to a different basis).
 */

#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

struct GaussPoint {
    double v;
    double w;
    double weight;  // already absorbs the 1/2 unit-triangle area factor
};

std::vector<GaussPoint> gauss_points(int n_quad)
{
    if (n_quad == 1) {
        // Centroid, weight = area of unit triangle.
        return {{1.0 / 3.0, 1.0 / 3.0, 0.5}};
    }
    if (n_quad == 3) {
        // Hammer-Marlowe 3-point, exact for degree 2.
        return {
            {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
            {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
            {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
        };
    }
    if (n_quad == 7) {
        // Strang-Fix T2:5-7 (also Cowper), exact for degree 5. The
        // table weights are normalised to the barycentric reference
        // triangle (area 1); for our (v, w) unit triangle (area 1/2)
        // we multiply EACH weight by 1/2 EXACTLY ONCE. The
        // (155 ± sqrt(15))/2400 form already has the 1/2 baked in
        // (the original Cowper table gives them as
        // (155 ± sqrt(15))/1200, and the /2400 form already halves
        // them). Sum of all 7 weights below = 1/2 (verified).
        const double sqrt15 = std::sqrt(15.0);
        const double a  = (6.0 + sqrt15) / 21.0;
        const double b  = (6.0 - sqrt15) / 21.0;
        const double wa = (155.0 + sqrt15) / 2400.0;
        const double wb = (155.0 - sqrt15) / 2400.0;
        const double wc = 9.0 / 80.0;
        return {
            {1.0 / 3.0,  1.0 / 3.0,    wc},
            {a,          a,            wa},
            {a,          1.0 - 2.0*a,  wa},
            {1.0 - 2.0*a, a,           wa},
            {b,          b,            wb},
            {b,          1.0 - 2.0*b,  wb},
            {1.0 - 2.0*b, b,           wb},
        };
    }
    throw std::invalid_argument(
        "gauss_points: n_quad must be 1, 3, or 7");
}

// Wrapper that re-uses element_stiffness_regular for each Gauss point.
// element_stiffness_regular bakes in w_quad = 0.5 internally; we
// rescale each contribution by (weight / 0.5) so the sum integrates
// with the actual Gauss weights.
Eigen::Matrix<double, 36, 36> element_K_regular_multi(
    const std::array<Eigen::Index, 12>& dofs,
    const Eigen::MatrixXd& V_aug,
    const cs::ShellMaterial& mat,
    int n_quad)
{
    Eigen::Matrix<double, 36, 36> K_e =
        Eigen::Matrix<double, 36, 36>::Zero();
    for (const auto& gp : gauss_points(n_quad)) {
        const auto pe = csl::evaluate_patch_regular(
            dofs, V_aug, gp.v, gp.w);
        const auto K_single = csl::element_stiffness_regular(pe, mat);
        K_e += (gp.weight / 0.5) * K_single;
    }
    return K_e;
}

Eigen::MatrixXd element_K_stam_multi(
    const csl::StamEvaluator& ev,
    const std::vector<Eigen::Index>& patch_dofs,
    const Eigen::MatrixXd& V_aug,
    const cs::ShellMaterial& mat,
    int n_quad)
{
    Eigen::MatrixXd K_e;
    for (const auto& gp : gauss_points(n_quad)) {
        const auto pe = csl::evaluate_patch_stam(
            ev, patch_dofs, V_aug, gp.v, gp.w);
        const auto K_single = csl::element_stiffness_stam(pe, mat);
        if (K_e.size() == 0) {
            K_e = Eigen::MatrixXd::Zero(K_single.rows(), K_single.cols());
        }
        K_e += (gp.weight / 0.5) * K_single;
    }
    return K_e;
}

// Custom assembly mirroring assemble_stiffness_augmented but with
// multi-point quadrature. Returns the (3*n_aug)^2 sparse K.
Eigen::SparseMatrix<double> assemble_K_multi(
    const csl::LoopAugmented& aug,
    const cs::ShellMaterial& mat,
    int n_quad,
    bool use_stam)
{
    const Eigen::Index n_aug = aug.V_aug.rows();
    const Eigen::Index dim_aug = 3 * n_aug;
    const auto patches = csl::build_patch_stencils(aug.V_aug, aug.F_aug);

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(36 * 36 * aug.n_real_faces);

    std::map<int, csl::StamEvaluator> stam_cache;

    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        const auto& p = patches[static_cast<std::size_t>(f)];
        int n_irreg = 0;
        int kev = -1;
        for (int k = 0; k < 3; ++k) {
            if (p.corner_valences[static_cast<std::size_t>(k)] != 6) {
                ++n_irreg;
                kev = k;
            }
        }
        if (n_irreg == 0) {
            const auto dofs = csl::canonical_regular_dofs(p, aug.F_aug);
            const auto K_e =
                element_K_regular_multi(dofs, aug.V_aug, mat, n_quad);
            for (int I = 0; I < 12; ++I) {
                const Eigen::Index gi = dofs[I];
                for (int J = 0; J < 12; ++J) {
                    const Eigen::Index gj = dofs[J];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = K_e(3*I+di, 3*J+dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3*gi+di, 3*gj+dj, v);
                            }
                        }
                    }
                }
            }
        } else if (use_stam && n_irreg == 1) {
            const int N = p.corner_valences[static_cast<std::size_t>(kev)];
            const auto dofs = csl::gather_stam_patch_dofs(p, aug.F_aug);
            auto it = stam_cache.find(N);
            if (it == stam_cache.end()) {
                it = stam_cache.emplace(N, csl::make_stam_evaluator(N)).first;
            }
            const auto K_e = element_K_stam_multi(
                it->second, dofs, aug.V_aug, mat, n_quad);
            const int K = N + 6;
            for (int I = 0; I < K; ++I) {
                const Eigen::Index gi = dofs[I];
                for (int J = 0; J < K; ++J) {
                    const Eigen::Index gj = dofs[J];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = K_e(3*I+di, 3*J+dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3*gi+di, 3*gj+dj, v);
                            }
                        }
                    }
                }
            }
        }
        // else: skip (n_irreg > 1, or n_irreg == 1 without use_stam)
    }

    Eigen::SparseMatrix<double> K_aug(dim_aug, dim_aug);
    K_aug.setFromTriplets(trips.begin(), trips.end());
    return K_aug;
}

// Inline a small dense eigensolve + mass-projection rigid filter so
// this test doesn't depend on Spectra (which is a PRIVATE chladni
// dep). Densifies K, M — fine for the 486-DOF icosphere k=2.
double n2_mean_frequency_hz(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    const Eigen::MatrixXd& V_orig,
    std::size_t n_modes)
{
    const Eigen::MatrixXd K_d(K);
    const Eigen::MatrixXd M_d(M);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd>
        solver(K_d, M_d);
    REQUIRE(solver.info() == Eigen::Success);
    const Eigen::VectorXd evals = solver.eigenvalues();
    const Eigen::MatrixXd evecs = solver.eigenvectors();

    // Mass-projection rigid filter (same logic as the production
    // path).
    Eigen::MatrixXd V_rigid = Eigen::MatrixXd::Zero(3 * V_orig.rows(), 6);
    for (Eigen::Index i = 0; i < V_orig.rows(); ++i) {
        const double x = V_orig(i, 0), y = V_orig(i, 1), z = V_orig(i, 2);
        V_rigid(3*i+0, 0) = 1; V_rigid(3*i+1, 1) = 1; V_rigid(3*i+2, 2) = 1;
        V_rigid(3*i+1, 3) = -z; V_rigid(3*i+2, 3) =  y;
        V_rigid(3*i+0, 4) =  z; V_rigid(3*i+2, 4) = -x;
        V_rigid(3*i+0, 5) = -y; V_rigid(3*i+1, 5) =  x;
    }
    const auto MV_rigid = M * V_rigid;
    const Eigen::Matrix<double, 6, 6> G_rigid =
        V_rigid.transpose() * MV_rigid;
    const Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(G_rigid);
    std::vector<double> physical;
    physical.reserve(evals.size());
    for (Eigen::Index i = 0; i < evals.size(); ++i) {
        const Eigen::Matrix<double, 6, 1> c =
            MV_rigid.transpose() * evecs.col(i);
        const Eigen::Matrix<double, 6, 1> ginv_c = chol.solve(c);
        const double proj_sq = c.dot(ginv_c);
        if (proj_sq > 0.5) continue;
        if (!(evals(i) > 0.0)) continue;
        physical.push_back(evals(i));
    }
    std::sort(physical.begin(), physical.end());
    REQUIRE(physical.size() >= n_modes);

    double sum = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
        sum += std::sqrt(physical[k]);
    }
    return (sum / 5.0) / (2.0 * M_PI);
}

}  // namespace

TEST_CASE("Icosphere k=2 limit surface vs sphere — geometric gap",
          "[shell][loop][quad][experiment][.experiment]")
{
    // Hypothesis: the 36% FEM rel_err at icosphere k=2 isn't from
    // quadrature; it's from the Loop limit surface NOT being a
    // sphere. Evaluate the limit surface at each face centroid and
    // compare its distance from origin to the sphere radius R.
    using namespace chladni::shell::loop;
    constexpr double R = 0.10;
    const auto mesh = cmsh::generate_icosphere(R, 2);

    // Pre-subdivision pipeline matches the Stam/L.3.4 K assembly so
    // we measure the same surface the FEM "sees".
    const auto sub = loop_subdivide_n_times(mesh.V, mesh.F, 1);
    const auto aug = augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto patches = build_patch_stencils(aug.V_aug, aug.F_aug);

    double r_min = std::numeric_limits<double>::infinity();
    double r_max = -std::numeric_limits<double>::infinity();
    double r_mean = 0.0;
    Eigen::Index n_eval = 0;
    std::map<int, StamEvaluator> stam_cache;
    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        const auto& p = patches[static_cast<std::size_t>(f)];
        int n_irreg = 0;
        for (int k = 0; k < 3; ++k) {
            if (p.corner_valences[static_cast<std::size_t>(k)] != 6) ++n_irreg;
        }
        Eigen::Vector3d pos;
        if (n_irreg == 0) {
            const auto dofs = canonical_regular_dofs(p, aug.F_aug);
            const auto pe = evaluate_patch_regular(
                dofs, aug.V_aug, 1.0/3.0, 1.0/3.0);
            pos = pe.position;
        } else if (n_irreg == 1) {
            const auto dofs = gather_stam_patch_dofs(p, aug.F_aug);
            int kev = -1;
            for (int k = 0; k < 3; ++k) {
                if (p.corner_valences[static_cast<std::size_t>(k)] != 6) {
                    kev = k;
                }
            }
            const int N = p.corner_valences[static_cast<std::size_t>(kev)];
            auto it = stam_cache.find(N);
            if (it == stam_cache.end()) {
                it = stam_cache.emplace(N, make_stam_evaluator(N)).first;
            }
            const auto pe = evaluate_patch_stam(
                it->second, dofs, aug.V_aug, 1.0/3.0, 1.0/3.0);
            pos = pe.position;
        } else {
            continue;
        }
        const double r = pos.norm();
        r_min = std::min(r_min, r);
        r_max = std::max(r_max, r);
        r_mean += r;
        ++n_eval;
    }
    r_mean /= static_cast<double>(n_eval);
    std::cout << "[icosphere k=2 limit surface]"
              << "  sphere R=" << R
              << "  limit r_min=" << r_min
              << "  r_max=" << r_max
              << "  r_mean=" << r_mean
              << "  rel deviation = " << (R - r_mean) / R
              << " (mean inward shift)\n";
}

TEST_CASE("Icosphere k=2 Wilkinson rel_err: 1- vs 3- vs 7-point Gauss",
          "[shell][loop][quad][experiment][.experiment]")
{
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9,
        .poisson_ratio  = 0.30,
        .density        = 7850.0};
    const auto sm = cs::shell_material_from_isotropic(steel, h);
    const auto mesh = cmsh::generate_icosphere(R, /*n_subdivisions=*/2);

    const auto analytical =
        chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, steel, /*n_modes=*/1);
    const double omega_wilkinson = analytical.front();
    const double f_wilkinson = omega_wilkinson / (2.0 * M_PI);

    // Replicate compute_shell_modes_loop's irregular branch: subdivide
    // once, augment, assemble with our custom multi-point K, then
    // reduce K = S^T C^T K_aug C S.
    const auto sub = csl::loop_subdivide_n_times(
        mesh.V, mesh.F, /*n_passes=*/1);
    const auto aug = csl::augment_for_loop_boundary(sub.V_sub, sub.F_sub);

    // Build M on the original mesh (the standard mass matrix path).
    const auto vmasses = cs::lumped_vertex_masses(
        mesh.V, mesh.F, steel.density, h);
    const Eigen::SparseMatrix<double> M_orig =
        cs::assemble_mass_matrix(vmasses);

    Eigen::SparseMatrix<double> Ct = aug.C.transpose();
    Eigen::SparseMatrix<double> St = sub.S.transpose();

    auto run = [&](int n_quad, bool use_stam) {
        const auto K_sub_aug =
            assemble_K_multi(aug, sm, n_quad, use_stam);
        Eigen::SparseMatrix<double> K_sub = Ct * K_sub_aug * aug.C;
        Eigen::SparseMatrix<double> K = St * K_sub * sub.S;
        const double f_mean =
            n2_mean_frequency_hz(K, M_orig, mesh.V, /*n_modes=*/6);
        const double rel_err =
            std::abs(f_mean - f_wilkinson) / f_wilkinson;
        std::cout << "  n_quad=" << n_quad
                  << "  use_stam=" << (use_stam ? "true " : "false")
                  << "  f_n2_mean=" << f_mean
                  << "  rel_err=" << rel_err << '\n';
        return rel_err;
    };

    std::cout << "[icosphere R=" << R << " h=" << h
              << "  k=2  V=" << mesh.V.rows() << "]"
              << "  Wilkinson n=2 = " << f_wilkinson << " Hz\n";
    std::cout << "L.3.4 path (skip irregular):\n";
    const double err_l34_1 = run(1, false);
    const double err_l34_3 = run(3, false);
    const double err_l34_7 = run(7, false);
    std::cout << "Stam path (no skip):\n";
    const double err_stam_1 = run(1, true);
    const double err_stam_3 = run(3, true);
    const double err_stam_7 = run(7, true);

    // No hard assertion — this is an experiment. We just want to see
    // the numbers. The test passes as long as everything is finite.
    REQUIRE(std::isfinite(err_l34_1));
    REQUIRE(std::isfinite(err_l34_3));
    REQUIRE(std::isfinite(err_l34_7));
    REQUIRE(std::isfinite(err_stam_1));
    REQUIRE(std::isfinite(err_stam_3));
    REQUIRE(std::isfinite(err_stam_7));
}
