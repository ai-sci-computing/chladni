/**
 * @file test_loop_global_K.cpp
 * @brief Global Loop-subdivision stiffness assembly tests (L.5b).
 *
 * Exercises @ref chladni::shell::loop::assemble_stiffness_augmented
 * and @ref chladni::shell::loop::assemble_stiffness_loop on a small
 * cylinder (n_around = 8, n_along = 2). Properties verified:
 *
 *  1. Shape: K_aug is square @f$3 n_\text{aug} \times 3 n_\text{aug}@f$;
 *     reduced K is square @f$3 n_\text{real} \times 3 n_\text{real}@f$.
 *  2. Symmetry: @f$K = K^\top@f$ on both K_aug and K (within numerical
 *     tolerance — round-off in the scatter and the sparse matrix
 *     product can introduce tiny asymmetry).
 *  3. Rigid-translation invariance: @f$K \cdot t = 0@f$ for the three
 *     standard unit translations, on both K_aug and K. Tests the
 *     element-level translation invariance (each @f$M^I, B^I@f$ in
 *     @ref element_stiffness_regular has zero column-sum) AND that the
 *     constraint matrix C correctly lifts a real-DOF translation to
 *     an augmented-DOF translation.
 *  4. Reduction sanity: @f$K = C^\top K_\text{aug} C@f$ — verify by
 *     computing both sides independently.
 *  5. Positive semi-definite: a few random non-rigid displacements
 *     yield non-negative @f$\tfrac12 u^\top K u@f$. (The full PSD
 *     check is left to validation in L.7 against analytic spectra.)
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <random>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

cmsh::TriMesh make_small_cylinder()
{
    return cmsh::generate_cylinder(1.0, 1.0, 8, 2);
}

cs::ShellMaterial make_material()
{
    cs::ShellMaterial m;
    m.k_L           = 1.0e6;
    m.k_B           = 1.0e3;
    m.poisson_ratio = 0.3;
    return m;
}

double frob_norm(const Eigen::SparseMatrix<double>& M)
{
    return Eigen::MatrixXd(M).norm();
}

bool sparse_almost_equal(const Eigen::SparseMatrix<double>& A,
                          const Eigen::SparseMatrix<double>& B,
                          double tol)
{
    return (Eigen::MatrixXd(A) - Eigen::MatrixXd(B)).cwiseAbs().maxCoeff()
           <= tol;
}

}  // namespace

TEST_CASE("assemble_stiffness_augmented: shape and symmetry on cylinder",
          "[shell][loop][global_k][shape]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto aug  = csl::augment_for_loop_boundary(mesh.V, mesh.F);
    const auto K_aug = csl::assemble_stiffness_augmented(aug, mat);

    REQUIRE(K_aug.rows() == 3 * (aug.n_real + aug.n_phantom));
    REQUIRE(K_aug.cols() == K_aug.rows());

    // Symmetry: K_aug == K_aug^T within tight tolerance. Tolerance
    // scales with the matrix's Frobenius norm because element K_e is
    // computed with multiplications that incur round-off.
    const double scale = frob_norm(K_aug);
    REQUIRE(scale > 0.0);
    Eigen::SparseMatrix<double> Kt = K_aug.transpose();
    Eigen::SparseMatrix<double> diff = K_aug - Kt;
    REQUIRE(frob_norm(diff) <= 1e-9 * scale);
}

TEST_CASE("assemble_stiffness_augmented: K_aug annihilates uniform translation in augmented DOFs",
          "[shell][loop][global_k][translation]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto aug  = csl::augment_for_loop_boundary(mesh.V, mesh.F);
    const auto K_aug = csl::assemble_stiffness_augmented(aug, mat);

    const Eigen::Index n_aug = aug.n_real + aug.n_phantom;
    // Uniform translation t in augmented DOF space: every vertex (real
    // or phantom) gets the same delta. K_aug should annihilate it
    // exactly because element_stiffness_regular's M^I and B^I have
    // zero row-sums in the spatial directions (Cirak-Ortiz Eq. 79-80
    // verify this once the basis sums to 1).
    const double scale = frob_norm(K_aug);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u_aug = Eigen::VectorXd::Zero(3 * n_aug);
        for (Eigen::Index v = 0; v < n_aug; ++v) {
            u_aug(3 * v + axis) = 1.0;
        }
        const Eigen::VectorXd Ku = K_aug * u_aug;
        // ||K_aug u|| should be vanishingly small relative to scale.
        REQUIRE(Ku.norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_loop: reduced K shape and symmetry",
          "[shell][loop][global_k][reduced]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto K = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat);

    REQUIRE(K.rows() == 3 * mesh.V.rows());
    REQUIRE(K.cols() == K.rows());

    const double scale = frob_norm(K);
    REQUIRE(scale > 0.0);
    Eigen::SparseMatrix<double> Kt = K.transpose();
    Eigen::SparseMatrix<double> diff = K - Kt;
    REQUIRE(frob_norm(diff) <= 1e-9 * scale);
}

TEST_CASE("assemble_stiffness_loop: rigid translation in real DOFs annihilates K",
          "[shell][loop][global_k][reduced][translation]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto K = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat);
    const Eigen::Index n_v = mesh.V.rows();
    const double scale = frob_norm(K);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * n_v);
        for (Eigen::Index v = 0; v < n_v; ++v) {
            u(3 * v + axis) = 1.0;
        }
        const Eigen::VectorXd Ku = K * u;
        REQUIRE(Ku.norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_loop: K matches C^T K_aug C exactly",
          "[shell][loop][global_k][reduction]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto aug   = csl::augment_for_loop_boundary(mesh.V, mesh.F);
    const auto K_aug = csl::assemble_stiffness_augmented(aug, mat);
    const auto K     = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat);

    Eigen::SparseMatrix<double> Ct = aug.C.transpose();
    Eigen::SparseMatrix<double> K_check = Ct * K_aug * aug.C;
    REQUIRE(sparse_almost_equal(K, K_check, 1e-12 * frob_norm(K)));
}

// ---------------------------------------------------------------------------
// L.3.4c: skip-irregular flag.
//
// Subdividing a closed octahedron (every vertex valence 4) produces a
// mesh on which 3 of every 4 sub-triangles per parent still have a
// valence-4 corner — the residual irregular sub-triangles surrounding
// each original extraordinary vertex. The skip flag silently drops
// their contribution; without it the assembler must throw.
// ---------------------------------------------------------------------------

namespace {

cmsh::TriMesh make_octahedron_mesh()
{
    cmsh::TriMesh m;
    m.V.resize(6, 3);
    m.V <<  1,  0,  0,
           -1,  0,  0,
            0,  1,  0,
            0, -1,  0,
            0,  0,  1,
            0,  0, -1;
    m.F.resize(8, 3);
    m.F << 0, 2, 4,
           2, 1, 4,
           1, 3, 4,
           3, 0, 4,
           2, 0, 5,
           1, 2, 5,
           3, 1, 5,
           0, 3, 5;
    return m;
}

}  // namespace

TEST_CASE("assemble_stiffness_loop: multi-pass changes K on irregular meshes only",
          "[shell][loop][global_k][multipass]")
{
    // Multi-pass subdivision matters only when the L.3.4 dispatch
    // fires, i.e. when at least one interior vertex is extraordinary.
    // On a regular cylinder (all interior valence 6) the fast path is
    // taken regardless of n_passes — K_1 == K_2.
    {
        const auto mesh = make_small_cylinder();
        const auto mat  = make_material();
        const auto K1 = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat, 1);
        const auto K2 = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat, 2);
        REQUIRE(sparse_almost_equal(K1, K2, 1e-12 * frob_norm(K1)));
    }

    // On the closed octahedron the irregular branch fires; n_passes=1
    // and n_passes=2 must produce different K (multi-pass actually
    // does something) but both must remain symmetric and translation-
    // invariant.
    {
        const auto m   = make_octahedron_mesh();
        const auto mat = make_material();
        const auto K1 = csl::assemble_stiffness_loop(m.V, m.F, mat, 1);
        const auto K2 = csl::assemble_stiffness_loop(m.V, m.F, mat, 2);

        REQUIRE(K1.rows() == K2.rows());

        // K1 != K2 — the second subdivision pass shrinks the dropped
        // residual irregular area from 25% to 6.25% per parent
        // triangle and pulls some of that energy back into K.
        const double scale = frob_norm(K1);
        REQUIRE(scale > 0.0);
        REQUIRE(frob_norm(K1 - K2) > 1e-3 * scale);

        // Both annihilate rigid translation along each axis.
        for (const auto& K : {K1, K2}) {
            const double s = frob_norm(K);
            Eigen::SparseMatrix<double> Kt = K.transpose();
            REQUIRE(frob_norm(K - Kt) <= 1e-9 * s);
            for (int axis = 0; axis < 3; ++axis) {
                Eigen::VectorXd t = Eigen::VectorXd::Zero(3 * m.V.rows());
                for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
                    t(3 * v + axis) = 1.0;
                }
                REQUIRE((K * t).norm() <= 1e-9 * s);
            }
        }
    }
}

TEST_CASE("assemble_stiffness_loop: throws on n_passes < 1",
          "[shell][loop][global_k][multipass][validation]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    REQUIRE_THROWS_AS(csl::assemble_stiffness_loop(m.V, m.F, mat, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(csl::assemble_stiffness_loop(m.V, m.F, mat, -1),
                      std::invalid_argument);
}

TEST_CASE("assemble_stiffness_loop: irregular path on closed octahedron — no throw",
          "[shell][loop][global_k][irregular]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();

    // Without L.3.4 wiring, this would throw on every parent triangle
    // because the octahedron's interior valences are all 4. With the
    // subdivision path, assembly succeeds.
    REQUIRE_NOTHROW(csl::assemble_stiffness_loop(m.V, m.F, mat));
}

TEST_CASE("assemble_stiffness_loop: irregular path — K shape, symmetry, translation",
          "[shell][loop][global_k][irregular]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    const auto K   = csl::assemble_stiffness_loop(m.V, m.F, mat);

    // Reduced K is on the original DOF layout (no S/C visible).
    REQUIRE(K.rows() == 3 * m.V.rows());
    REQUIRE(K.cols() == K.rows());

    const double scale = frob_norm(K);
    REQUIRE(scale > 0.0);

    // Symmetry.
    Eigen::SparseMatrix<double> Kt = K.transpose();
    REQUIRE(frob_norm(K - Kt) <= 1e-9 * scale);

    // Rigid translation along each axis annihilates K. The S * (rigid
    // translation in real DOFs) = (rigid translation in subdivided DOFs)
    // identity is the L.3.4a "S preserves constants" property; the
    // Schweitzer phantom rule then preserves it through C; finally the
    // element K_e annihilates rigid translation. So all three axes must
    // satisfy K * t_axis = 0.
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * m.V.rows());
        for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
            u(3 * v + axis) = 1.0;
        }
        REQUIRE((K * u).norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_loop: irregular path — PSD on random non-rigid u",
          "[shell][loop][global_k][irregular][psd]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    const auto K   = csl::assemble_stiffness_loop(m.V, m.F, mat);
    const Eigen::Index n_v = m.V.rows();

    std::mt19937 rng(20260509);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int trial = 0; trial < 8; ++trial) {
        Eigen::VectorXd u(3 * n_v);
        for (Eigen::Index i = 0; i < u.size(); ++i) u(i) = uni(rng);
        for (int axis = 0; axis < 3; ++axis) {
            double mean = 0.0;
            for (Eigen::Index v = 0; v < n_v; ++v) mean += u(3 * v + axis);
            mean /= static_cast<double>(n_v);
            for (Eigen::Index v = 0; v < n_v; ++v) u(3 * v + axis) -= mean;
        }
        const double q = u.dot(K * u);
        REQUIRE(q >= -1e-10 * frob_norm(K));
        REQUIRE(q > 0.0);
    }
}

TEST_CASE("assemble_stiffness_augmented: throws on extraordinary corner by default",
          "[shell][loop][global_k][skip]")
{
    const auto m = make_octahedron_mesh();
    const auto sub = csl::loop_subdivide_one_step(m.V, m.F);
    const auto aug = csl::augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto mat = make_material();

    REQUIRE_THROWS_AS(
        csl::assemble_stiffness_augmented(aug, mat),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        csl::assemble_stiffness_augmented(aug, mat, /*skip=*/false),
        std::runtime_error);
}

TEST_CASE("assemble_stiffness_augmented: skip flag returns valid K on irregular mesh",
          "[shell][loop][global_k][skip]")
{
    const auto m = make_octahedron_mesh();
    const auto sub = csl::loop_subdivide_one_step(m.V, m.F);
    const auto aug = csl::augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto mat = make_material();

    const auto K_aug = csl::assemble_stiffness_augmented(
        aug, mat, /*skip_irregular_triangles=*/true);

    const Eigen::Index n_aug = aug.n_real + aug.n_phantom;
    REQUIRE(K_aug.rows() == 3 * n_aug);
    REQUIRE(K_aug.cols() == K_aug.rows());

    // Symmetry.
    const double scale = frob_norm(K_aug);
    REQUIRE(scale > 0.0);
    Eigen::SparseMatrix<double> Kt = K_aug.transpose();
    Eigen::SparseMatrix<double> diff = K_aug - Kt;
    REQUIRE(frob_norm(diff) <= 1e-9 * scale);

    // Rigid translation along each axis annihilates K_aug — element_K
    // is translation-invariant per Cirak-Ortiz Eq. 79-80, so dropping
    // some triangles cannot break this property.
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * n_aug);
        for (Eigen::Index v = 0; v < n_aug; ++v) {
            u(3 * v + axis) = 1.0;
        }
        REQUIRE((K_aug * u).norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_loop: K is non-negative on random non-rigid displacements",
          "[shell][loop][global_k][psd]")
{
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto K = csl::assemble_stiffness_loop(mesh.V, mesh.F, mat);
    const Eigen::Index n_v = mesh.V.rows();

    std::mt19937 rng(20260509);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (int trial = 0; trial < 8; ++trial) {
        Eigen::VectorXd u(3 * n_v);
        for (Eigen::Index i = 0; i < u.size(); ++i) u(i) = uni(rng);
        // Project out the 3 rigid-body translations (axis-aligned).
        for (int axis = 0; axis < 3; ++axis) {
            double mean = 0.0;
            for (Eigen::Index v = 0; v < n_v; ++v) mean += u(3 * v + axis);
            mean /= static_cast<double>(n_v);
            for (Eigen::Index v = 0; v < n_v; ++v) u(3 * v + axis) -= mean;
        }
        const double q = u.dot(K * u);
        // K is PSD; q must be > 0 for non-zero u with rigid
        // translations removed (provided u is not a rigid rotation,
        // which random sampling makes vanishingly unlikely).
        REQUIRE(q >= -1e-10 * frob_norm(K));
        REQUIRE(q > 0.0);
    }
}

// ---------------------------------------------------------------------------
// S.7c: Stam path through assemble_stiffness_augmented and
// assemble_stiffness_loop.
//
// With use_stam_for_irregular=true (resp. use_stam=true) the residual
// irregular sub-triangles around extraordinary vertices are evaluated
// exactly via the Stam 1999 eigenbasis machinery, replacing the L.3.4
// step-1 drop approximation. K must remain symmetric and translation-
// invariant, and must differ from the L.3.4 result (the irregular
// contribution is no longer dropped).
// ---------------------------------------------------------------------------

TEST_CASE("assemble_stiffness_augmented: Stam path on subdivided octahedron — shape and translation",
          "[shell][loop][global_k][stam]")
{
    const auto m   = make_octahedron_mesh();
    const auto sub = csl::loop_subdivide_one_step(m.V, m.F);
    const auto aug = csl::augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto mat = make_material();

    const auto K_aug = csl::assemble_stiffness_augmented(
        aug, mat,
        /*skip_irregular_triangles=*/false,
        /*use_stam_for_irregular=*/true);

    const Eigen::Index n_aug = aug.n_real + aug.n_phantom;
    REQUIRE(K_aug.rows() == 3 * n_aug);
    REQUIRE(K_aug.cols() == K_aug.rows());

    const double scale = frob_norm(K_aug);
    REQUIRE(scale > 0.0);

    // Symmetry.
    Eigen::SparseMatrix<double> Kt = K_aug.transpose();
    REQUIRE(frob_norm(K_aug - Kt) <= 1e-9 * scale);

    // Rigid translation along each axis annihilates K_aug. The Stam
    // element K_e is translation-invariant by the same argument as the
    // regular element K_e (Phi_0 = 1 means uniform shifts of all K
    // patch positions produce zero strain).
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * n_aug);
        for (Eigen::Index v = 0; v < n_aug; ++v) {
            u(3 * v + axis) = 1.0;
        }
        REQUIRE((K_aug * u).norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_augmented: Stam path differs from skip path on octahedron",
          "[shell][loop][global_k][stam]")
{
    // Same subdivided mesh, two policies. The Stam path adds the
    // residual irregular-sub-triangle energy that the skip path drops,
    // so the two K_aug must differ — and the difference must be
    // non-trivial (more than round-off).
    const auto m   = make_octahedron_mesh();
    const auto sub = csl::loop_subdivide_one_step(m.V, m.F);
    const auto aug = csl::augment_for_loop_boundary(sub.V_sub, sub.F_sub);
    const auto mat = make_material();

    const auto K_skip = csl::assemble_stiffness_augmented(
        aug, mat,
        /*skip_irregular_triangles=*/true,
        /*use_stam_for_irregular=*/false);
    const auto K_stam = csl::assemble_stiffness_augmented(
        aug, mat,
        /*skip_irregular_triangles=*/false,
        /*use_stam_for_irregular=*/true);

    REQUIRE(K_skip.rows() == K_stam.rows());
    const double s = frob_norm(K_skip);
    REQUIRE(s > 0.0);
    REQUIRE(frob_norm(K_stam - K_skip) > 1e-3 * s);
}

TEST_CASE("assemble_stiffness_loop: use_stam=true on regular mesh equals fast path",
          "[shell][loop][global_k][stam]")
{
    // No extraordinary interior vertices on the small cylinder, so
    // use_stam should be a no-op (fast path is taken regardless).
    const auto mesh = make_small_cylinder();
    const auto mat  = make_material();
    const auto K_default = csl::assemble_stiffness_loop(
        mesh.V, mesh.F, mat);
    const auto K_stam    = csl::assemble_stiffness_loop(
        mesh.V, mesh.F, mat, /*n_passes=*/1, /*use_stam=*/true);
    REQUIRE(sparse_almost_equal(
        K_default, K_stam, 1e-12 * frob_norm(K_default)));
}

TEST_CASE("assemble_stiffness_loop: use_stam=true on octahedron — shape, symmetry, translation",
          "[shell][loop][global_k][stam]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    const auto K   = csl::assemble_stiffness_loop(
        m.V, m.F, mat, /*n_passes=*/1, /*use_stam=*/true);

    REQUIRE(K.rows() == 3 * m.V.rows());
    REQUIRE(K.cols() == K.rows());

    const double scale = frob_norm(K);
    REQUIRE(scale > 0.0);

    Eigen::SparseMatrix<double> Kt = K.transpose();
    REQUIRE(frob_norm(K - Kt) <= 1e-9 * scale);

    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * m.V.rows());
        for (Eigen::Index v = 0; v < m.V.rows(); ++v) {
            u(3 * v + axis) = 1.0;
        }
        REQUIRE((K * u).norm() <= 1e-9 * scale);
    }
}

TEST_CASE("assemble_stiffness_loop: Stam K differs from L.3.4 K on octahedron",
          "[shell][loop][global_k][stam]")
{
    // The Stam path adds the residual irregular energy that the L.3.4
    // skip drops, so the reduced K's must differ. The S transform and
    // C reduction are linear and shared between paths, so the
    // difference propagates straight through.
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    const auto K_l34  = csl::assemble_stiffness_loop(
        m.V, m.F, mat, /*n_passes=*/1, /*use_stam=*/false);
    const auto K_stam = csl::assemble_stiffness_loop(
        m.V, m.F, mat, /*n_passes=*/1, /*use_stam=*/true);

    REQUIRE(K_l34.rows() == K_stam.rows());
    const double s = frob_norm(K_l34);
    REQUIRE(s > 0.0);
    REQUIRE(frob_norm(K_stam - K_l34) > 1e-3 * s);
}

TEST_CASE("assemble_stiffness_loop: Stam K is PSD on octahedron",
          "[shell][loop][global_k][stam][psd]")
{
    const auto m   = make_octahedron_mesh();
    const auto mat = make_material();
    const auto K   = csl::assemble_stiffness_loop(
        m.V, m.F, mat, /*n_passes=*/1, /*use_stam=*/true);
    const Eigen::Index n_v = m.V.rows();

    std::mt19937 rng(20260512);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int trial = 0; trial < 8; ++trial) {
        Eigen::VectorXd u(3 * n_v);
        for (Eigen::Index i = 0; i < u.size(); ++i) u(i) = uni(rng);
        for (int axis = 0; axis < 3; ++axis) {
            double mean = 0.0;
            for (Eigen::Index v = 0; v < n_v; ++v) mean += u(3 * v + axis);
            mean /= static_cast<double>(n_v);
            for (Eigen::Index v = 0; v < n_v; ++v) u(3 * v + axis) -= mean;
        }
        const double q = u.dot(K * u);
        REQUIRE(q >= -1e-10 * frob_norm(K));
        REQUIRE(q > 0.0);
    }
}
