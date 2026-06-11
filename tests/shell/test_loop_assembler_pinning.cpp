/**
 * @file test_loop_assembler_pinning.cpp
 * @brief Pin the LoopAssembler refactor as a bit-identical wrapper.
 *
 * The Step-3/Step-4 refactor introduced an abstract
 * @ref chladni::shell::ShellAssembler interface plus a concrete
 * @ref chladni::shell::LoopAssembler that wraps the existing
 * @ref chladni::shell::loop::assemble_stiffness_loop /
 * @ref chladni::shell::loop::assemble_mass_loop free-function pipeline.
 * The wrapper exposes a @ref chladni::shell::MassLumping policy on
 * top of the consistent Galerkin assembly; the row-sum lump previously
 * applied unconditionally inside @ref chladni::shell::compute_shell_modes_loop
 * is now an opt-in choice (the default flipped to consistent on
 * 2026-05-17 late after the lumping bias was diagnosed; see the
 * @c m_lump docstring in assembler.hpp for context).
 *
 * This test pins that contract: with default
 * @ref LoopAssembler::Params (Stam on, n_passes=1, 7-pt Dunavant K/M,
 * @ref MassLumping::None), @c LoopAssembler::assemble_K returns exactly
 * the matrix produced by @c assemble_stiffness_loop. The two
 * @ref MassLumping cases (None passes through @c assemble_mass_loop
 * unchanged; RowSum applies the row-sum lump) are each pinned by a
 * separate test case below. Two fixtures cover the regular fast path
 * (icosphere k=2 projected to a sphere via the bundled generator) and
 * the irregular subdivision path (polar disk with a central
 * extraordinary vertex).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <limits>
#include <stdexcept>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

cs::ShellMaterial make_material()
{
    cs::ShellMaterial sm;
    sm.k_L           = 1.0e6;
    sm.k_B           = 1.0e3;
    sm.poisson_ratio = 0.3;
    return sm;
}

constexpr double k_surface_density = 2.5;  // kg/m^2 — rho * h, arbitrary > 0.

Eigen::SparseMatrix<double> row_sum_lump_reference(
    const Eigen::SparseMatrix<double>& M_full)
{
    const Eigen::VectorXd row_sums =
        M_full * Eigen::VectorXd::Ones(M_full.cols());

    Eigen::SparseMatrix<double> M_lumped(M_full.rows(), M_full.cols());
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(M_full.rows()));
    for (Eigen::Index i = 0; i < M_full.rows(); ++i) {
        const double m_ii = row_sums(i);
        if (m_ii > 0.0) {
            trips.emplace_back(i, i, m_ii);
        }
    }
    M_lumped.setFromTriplets(trips.begin(), trips.end());
    M_lumped.makeCompressed();
    return M_lumped;
}

bool sparse_exactly_equal(const Eigen::SparseMatrix<double>& A,
                          const Eigen::SparseMatrix<double>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    const Eigen::MatrixXd D = Eigen::MatrixXd(A) - Eigen::MatrixXd(B);
    // Exact bit-identity is too brittle once Eigen reorders triplets,
    // but the wrapper performs no arithmetic beyond row-sum on M, so a
    // very tight tolerance (relative to the matrix scale) suffices.
    const double scale = std::max(
        Eigen::MatrixXd(A).cwiseAbs().maxCoeff(),
        Eigen::MatrixXd(B).cwiseAbs().maxCoeff());
    return D.cwiseAbs().maxCoeff() <= 1e-12 * std::max(scale, 1.0);
}

}  // namespace

TEST_CASE("LoopAssembler: defaults wrap assemble_stiffness_loop on regular mesh",
          "[shell][loop][assembler][pinning]")
{
    // Icosphere k=2 — all interior vertices valence-6, fast path.
    const auto mesh = cmsh::generate_icosphere(0.10, 2);
    const auto sm   = make_material();

    cs::LoopAssembler::Params p;  // shipped defaults
    cs::LoopAssembler asm_default{p};

    const auto K_via_assembler = asm_default.assemble_K(mesh.V, mesh.F, sm);
    const auto K_via_free      = csl::assemble_stiffness_loop(
        mesh.V, mesh.F, sm, p.n_passes, p.use_stam);

    REQUIRE(sparse_exactly_equal(K_via_assembler, K_via_free));
}

TEST_CASE("LoopAssembler: MassLumping::RowSum lumps assemble_mass_loop on regular mesh",
          "[shell][loop][assembler][pinning]")
{
    const auto mesh = cmsh::generate_icosphere(0.10, 2);

    cs::LoopAssembler::Params p;
    p.m_lump = cs::MassLumping::RowSum;
    cs::LoopAssembler asm_lumped{p};

    const auto M_via_assembler =
        asm_lumped.assemble_M(mesh.V, mesh.F, k_surface_density);

    const auto M_full = csl::assemble_mass_loop(
        mesh.V, mesh.F, k_surface_density, p.n_passes, p.use_stam);
    const auto M_lumped_ref = row_sum_lump_reference(M_full);

    REQUIRE(sparse_exactly_equal(M_via_assembler, M_lumped_ref));
}

TEST_CASE("LoopAssembler: defaults wrap assemble_stiffness_loop on irregular mesh",
          "[shell][loop][assembler][pinning]")
{
    // Polar disk has one extraordinary central vertex (valence =
    // n_azimuthal). Exercises the subdivision branch.
    const auto mesh = cmsh::generate_circular_disk(0.10, 12, 4);
    const auto sm   = make_material();

    cs::LoopAssembler::Params p;  // shipped defaults — Stam on, n_passes=1
    cs::LoopAssembler asm_default{p};

    const auto K_via_assembler = asm_default.assemble_K(mesh.V, mesh.F, sm);
    const auto K_via_free      = csl::assemble_stiffness_loop(
        mesh.V, mesh.F, sm, p.n_passes, p.use_stam);

    REQUIRE(sparse_exactly_equal(K_via_assembler, K_via_free));
}

TEST_CASE("LoopAssembler: MassLumping::RowSum lumps assemble_mass_loop on irregular mesh",
          "[shell][loop][assembler][pinning]")
{
    const auto mesh = cmsh::generate_circular_disk(0.10, 12, 4);

    cs::LoopAssembler::Params p;
    p.m_lump = cs::MassLumping::RowSum;
    cs::LoopAssembler asm_lumped{p};

    const auto M_via_assembler =
        asm_lumped.assemble_M(mesh.V, mesh.F, k_surface_density);

    const auto M_full = csl::assemble_mass_loop(
        mesh.V, mesh.F, k_surface_density, p.n_passes, p.use_stam);
    const auto M_lumped_ref = row_sum_lump_reference(M_full);

    REQUIRE(sparse_exactly_equal(M_via_assembler, M_lumped_ref));
}

TEST_CASE("LoopAssembler: MassLumping::None returns the consistent mass matrix",
          "[shell][loop][assembler][pinning]")
{
    const auto mesh = cmsh::generate_icosphere(0.10, 2);

    cs::LoopAssembler::Params p;
    p.m_lump = cs::MassLumping::None;
    cs::LoopAssembler asm_consistent{p};

    const auto M_via_assembler =
        asm_consistent.assemble_M(mesh.V, mesh.F, k_surface_density);

    const auto M_full = csl::assemble_mass_loop(
        mesh.V, mesh.F, k_surface_density, p.n_passes, p.use_stam);

    REQUIRE(sparse_exactly_equal(M_via_assembler, M_full));
}

TEST_CASE("LoopAssembler: lower-order quadrature rules produce valid K / M",
          "[shell][loop][assembler][pinning]")
{
    // The 1-pt centroid and 3-pt edge-midpoint rules are now wired
    // through every kernel. They produce *different* K / M than the
    // 7-pt Dunavant default (lower-order under-integration of the
    // degree-4..8 integrands) but the matrices must still be
    // structurally valid: same shape, same DOF layout, symmetric,
    // and not identically zero. This test guards against silent
    // breakage of the new dispatch — exact numerical comparison vs
    // analytic spectra lives in the dedicated quadrature-A/B test.
    const auto mesh = cmsh::generate_icosphere(0.10, 1);
    const auto sm   = make_material();

    cs::LoopAssembler::Params p_default;
    cs::LoopAssembler asm_default{p_default};
    const auto K_7pt = asm_default.assemble_K(mesh.V, mesh.F, sm);
    const auto M_7pt = asm_default.assemble_M(
        mesh.V, mesh.F, k_surface_density);

    for (const auto rule : {cs::QuadratureRule::OnePointCentroid,
                            cs::QuadratureRule::ThreePointEdgeMid}) {
        cs::LoopAssembler::Params p;
        p.k_quad = rule;
        p.m_quad = rule;
        cs::LoopAssembler asm_lo{p};

        const auto K_lo = asm_lo.assemble_K(mesh.V, mesh.F, sm);
        const auto M_lo = asm_lo.assemble_M(
            mesh.V, mesh.F, k_surface_density);

        REQUIRE(K_lo.rows() == K_7pt.rows());
        REQUIRE(K_lo.cols() == K_7pt.cols());
        REQUIRE(M_lo.rows() == M_7pt.rows());
        REQUIRE(M_lo.cols() == M_7pt.cols());

        const Eigen::MatrixXd K_dense = K_lo;
        const Eigen::MatrixXd M_dense = M_lo;
        REQUIRE(K_dense.norm() > 0.0);
        REQUIRE(M_dense.norm() > 0.0);

        // Lower-order rules ARE different from 7-pt — this is the
        // whole point of exposing them as A/B knobs.
        const double k_diff = (K_dense - Eigen::MatrixXd(K_7pt))
            .cwiseAbs().maxCoeff();
        REQUIRE(k_diff > 1e-6 * K_dense.cwiseAbs().maxCoeff());

        // Symmetry of K.
        const Eigen::MatrixXd k_sym_err = K_dense - K_dense.transpose();
        REQUIRE(k_sym_err.cwiseAbs().maxCoeff()
                <= 1e-9 * K_dense.cwiseAbs().maxCoeff());
    }
}

TEST_CASE("LoopAssembler: rejects n_passes < 1 at construction",
          "[shell][loop][assembler][pinning]")
{
    cs::LoopAssembler::Params p;
    p.n_passes = 0;
    REQUIRE_THROWS_AS(cs::LoopAssembler{p}, std::invalid_argument);
}

TEST_CASE("LoopAssembler: shipped Params defaults are 7-pt Dunavant + consistent mass "
          "(review7 I10)",
          "[shell][loop][assembler][pinning]")
{
    // The README's headline Loop accuracy column (6.52/7.03/3.54 %) and the
    // benchmark table are valid ONLY for these defaults. The disk-table test
    // that produces those numbers is benchmark-gated (out of the default
    // suite), so without this pin a silent flip of the shipped default would
    // change the documented numbers with zero failing tests — exactly the
    // drift that left method_comparison_matrix.md saying "3pt" (review7 I4).
    const cs::LoopAssembler::Params p;
    REQUIRE(p.k_quad == cs::QuadratureRule::SevenPointDunavant);
    REQUIRE(p.m_quad == cs::QuadratureRule::SevenPointDunavant);
    REQUIRE(p.m_lump == cs::MassLumping::None);
}

TEST_CASE("assemble_stiffness_loop rejects non-finite vertex geometry "
          "(review5 G1)",
          "[shell][loop][assembler]")
{
    // The stiffness kernels guarded only `a_det <= 0.0`, which a NaN
    // a_det slips through (NaN <= 0 is false), so sqrt(NaN) propagated
    // into K with no allFinite backstop on the Loop path. Both the
    // regular kernel (flat plate, valence-6 interior -> fast path) and
    // the Stam kernel (disk, central extraordinary vertex -> subdivision)
    // must now throw on a non-finite vertex.
    const auto sm  = make_material();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    SECTION("regular kernel (fast path)")
    {
        auto plate = cmsh::generate_flat_plate(0.1, 0.1, 4, 4);
        plate.V(0, 2) = nan;
        REQUIRE_THROWS(csl::assemble_stiffness_loop(
            plate.V, plate.F, sm, /*n_passes=*/1, /*use_stam=*/false));
    }
    SECTION("Stam kernel (subdivision path)")
    {
        auto disk = cmsh::generate_circular_disk(0.10, 12, 4);
        disk.V(0, 2) = nan;
        REQUIRE_THROWS(csl::assemble_stiffness_loop(
            disk.V, disk.F, sm, /*n_passes=*/1, /*use_stam=*/true));
    }
}

TEST_CASE("assemble_mass_loop rejects non-finite vertex geometry "
          "(review6 H1)",
          "[shell][loop][assembler]")
{
    // The mass-side twin of review5 G1: assemble_mass_loop returned its
    // Ct*M*C / St*M*S products directly with no allFinite backstop, while
    // its stiffness twin gained one in the G1 remediation. A non-finite M
    // flows unguarded into the rigid-filtered eigensolver. Both the
    // regular (fast) and Stam (subdivision) paths must throw on a
    // non-finite vertex, matching assemble_stiffness_loop.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double density = 1.0;

    SECTION("regular kernel (fast path)")
    {
        auto plate = cmsh::generate_flat_plate(0.1, 0.1, 4, 4);
        plate.V(0, 2) = nan;
        REQUIRE_THROWS(csl::assemble_mass_loop(
            plate.V, plate.F, density, /*n_passes=*/1, /*use_stam=*/false));
    }
    SECTION("Stam kernel (subdivision path)")
    {
        auto disk = cmsh::generate_circular_disk(0.10, 12, 4);
        disk.V(0, 2) = nan;
        REQUIRE_THROWS(csl::assemble_mass_loop(
            disk.V, disk.F, density, /*n_passes=*/1, /*use_stam=*/true));
    }
}
