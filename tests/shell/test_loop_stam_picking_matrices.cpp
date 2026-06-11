/**
 * @file test_loop_stam_picking_matrices.cpp
 * @brief S.3b — picking matrices P_1, P_2, P_3 for Stam Eq. (2).
 *
 * Pins
 * @ref chladni::shell::loop::stam_picking_matrices to the algebraic
 * identities that the picking machinery must satisfy:
 *
 *  - Each @f$ P_k @f$ is shape @f$ 12 \times (N + 12) @f$ and is a 0/1
 *    selection matrix with exactly one @c 1 per row.
 *  - The 36 selected (tile, slot) -> Stam-M entries cover at most
 *    @c N + 12 distinct Stam-M indices (overlap on shared
 *    edge-opposite vertices between adjacent regular sub-tiles).
 *  - End-to-end: with arbitrary scalar control values @c c at the
 *    @c K = N + 6 input vertices, picking via @f$ P_k \bar A c @f$ on
 *    each tile reproduces the *direct* subdivide-then-read-canonical
 *    DOF values from one Loop-subdivision step on the same patch.
 *    This is the strongest pin: it ties together
 *    @ref build_extended_subdivision_matrix_bar (Stam App. B blocks),
 *    the V_sub -> Stam-M index correspondence, and
 *    @ref canonical_regular_dofs.
 *  - Input validation: N < 3 throws.
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

namespace {

/// Mirror of @c build_irregular_patch in @c loop_stam.cpp — needed by the
/// end-to-end test so we can re-run the same subdivision step ourselves
/// and read out the canonical DOFs directly.
///
/// Keeping the geometry in lockstep with the implementation is by design:
/// the picking matrices are indexed for the patch produced by
/// @c stam_picking_matrices' internal helper, so any divergence here
/// would invalidate the comparison and the test would fail loudly.
void build_irregular_patch(int N,
                           Eigen::MatrixXd& V,
                           Eigen::MatrixXi& F)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int K = N + 6;
    V.resize(K, 3);
    V.row(0).setZero();
    for (int k = 1; k <= N; ++k) {
        const double angle = -2.0 * pi * static_cast<double>(k - 1)
                                       / static_cast<double>(N);
        V.row(k) << std::cos(angle), std::sin(angle), 0.0;
    }
    constexpr double R_outer = 2.0;
    auto place = [&](int idx, double angle) {
        V.row(idx) << R_outer * std::cos(angle),
                      R_outer * std::sin(angle),
                      0.0;
    };
    const double Nf = static_cast<double>(N);
    place(N + 1,  pi / Nf);
    place(N + 2, -pi / (2.0 * Nf));
    place(N + 3, -pi / Nf);
    place(N + 4,  2.0 * pi / Nf + pi / (2.0 * Nf));
    place(N + 5,  3.0 * pi / Nf);

    F.resize(N + 7, 3);
    F.row(0) << 0, 1, N;
    for (int k = 1; k <= N - 1; ++k) {
        F.row(k) << 0, k + 1, k;
    }
    F.row(N + 0) << 1, N + 1, N;
    F.row(N + 1) << 1, N + 2, N + 1;
    F.row(N + 2) << 1, N + 3, N + 2;
    F.row(N + 3) << 1, 2,     N + 3;
    F.row(N + 4) << N, N + 1, N + 4;
    F.row(N + 5) << N, N + 4, N + 5;
    F.row(N + 6) << N, N + 5, N - 1;
}

}  // namespace

TEST_CASE("stam_picking_matrices: shape and 0/1 row-sum-one structure",
          "[shell][loop][stam][picking]")
{
    using chladni::shell::loop::stam_picking_matrices;

    for (int N : {3, 4, 5, 6, 7, 8, 12, 16}) {
        CAPTURE(N);
        const auto picks = stam_picking_matrices(N);
        REQUIRE(picks.N == N);
        const int M = N + 12;

        for (int k = 0; k < 3; ++k) {
            CAPTURE(k);
            const auto& P = picks.P[static_cast<std::size_t>(k)];

            REQUIRE(P.rows() == 12);
            REQUIRE(P.cols() == M);

            int nonzeros = 0;
            for (int slot = 0; slot < 12; ++slot) {
                CAPTURE(slot);
                int slot_nz = 0;
                for (int j = 0; j < M; ++j) {
                    const double v = P(slot, j);
                    REQUIRE((v == 0.0 || v == 1.0));
                    if (v == 1.0) ++slot_nz;
                }
                // Exactly one Stam vertex picked per slot.
                REQUIRE(slot_nz == 1);
                nonzeros += slot_nz;
            }
            REQUIRE(nonzeros == 12);
        }
    }
}

TEST_CASE("stam_picking_matrices: shared-vertex column coverage across tiles",
          "[shell][loop][stam][picking]")
{
    using chladni::shell::loop::stam_picking_matrices;

    // Coverage characterisation:
    //  - Total picks = 3 tiles * 12 slots = 36.
    //  - Per-tile col_sum: at most 1 for N >= 4. N=3 is degenerate
    //    (vertex 2 doubles as N-1 so m_{0,2} = m_{0,N-1} appears in
    //    BOTH m_{0,1}'s and m_{0,N}'s outer rings on the medial sub-tile;
    //    that Stam index is picked twice within the medial tile and up to
    //    four times across all three tiles).
    //  - "Middle" spoke midpoints m_{0,k} for k in [3, N-2] (Stam
    //    indices 3..N-2) are NOT needed by any of the 3 regular
    //    sub-tiles — they belong to the new patch's 1-ring around the
    //    extraordinary vertex and are consumed by the *recursion* on the
    //    irregular sub-tile (handled by the Stam exact-eval machinery,
    //    not by direct box-spline evaluation at this level). For N <= 4
    //    that range is empty; for N >= 5 it is exactly Stam 3..N-2.
    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const auto picks = stam_picking_matrices(N);
        const int M = N + 12;
        const int max_per_tile = (N == 3 ? 2 : 1);
        const int max_across   = (N == 3 ? 4 : 3);

        Eigen::VectorXi col_sum_total = Eigen::VectorXi::Zero(M);
        for (int k = 0; k < 3; ++k) {
            const auto& P = picks.P[static_cast<std::size_t>(k)];
            for (int j = 0; j < M; ++j) {
                int col_sum = 0;
                for (int slot = 0; slot < 12; ++slot) {
                    if (P(slot, j) == 1.0) ++col_sum;
                }
                CAPTURE(k, j);
                REQUIRE(col_sum <= max_per_tile);
                col_sum_total(j) += col_sum;
            }
        }
        REQUIRE(col_sum_total.sum() == 36);
        REQUIRE(col_sum_total.maxCoeff() <= max_across);

        // Indices that MUST be covered (the 16-vertex box-spline
        // neighbourhood spanned by the 3 regular sub-tiles).
        std::vector<int> covered_indices = {
            0, 1, 2, N - 1, N, N + 1, N + 2, N + 3, N + 4, N + 5,
            N + 6, N + 7, N + 8, N + 9, N + 10, N + 11
        };
        for (int idx : covered_indices) {
            CAPTURE(idx);
            REQUIRE(col_sum_total(idx) >= 1);
        }
        // Indices that MUST be uncovered: the "middle" spoke midpoints
        // m_{0,k} for k in [3, N-2]. Empty for N <= 4.
        for (int idx = 3; idx <= N - 2; ++idx) {
            CAPTURE(idx);
            REQUIRE(col_sum_total(idx) == 0);
        }
    }
}

TEST_CASE("stam_picking_matrices: end-to-end vs direct subdivision",
          "[shell][loop][stam][picking]")
{
    using chladni::shell::loop::build_extended_subdivision_matrix_bar;
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::canonical_regular_dofs;
    using chladni::shell::loop::loop_subdivide_one_step;
    using chladni::shell::loop::stam_picking_matrices;

    // tile k -> regular sub-face row index in F_sub for face 0 = central
    // triangle; matches the dispatch in stam_picking_matrices.
    constexpr std::array<Eigen::Index, 3> sub_face_idx = {1, 3, 2};

    for (int N : {3, 4, 5, 6, 7, 8, 12}) {
        CAPTURE(N);
        const int K = N + 6;
        const auto picks = stam_picking_matrices(N);
        const Eigen::MatrixXd Abar
            = build_extended_subdivision_matrix_bar(N);

        // Build the same patch the implementation uses, but stuff a
        // distinctive scalar control value into the z-component while
        // keeping the geometric (x, y) layout (so subdivide and the
        // patch-stencil walks see a valid manifold mesh).
        Eigen::MatrixXd V_geom;
        Eigen::MatrixXi F;
        build_irregular_patch(N, V_geom, F);
        Eigen::VectorXd c0(K);
        for (int i = 0; i < K; ++i) {
            c0(i) = std::sin(0.7 * (i + 1)) + 0.31 * std::cos(1.13 * (i + 2));
        }
        Eigen::MatrixXd V_test = V_geom;
        V_test.col(2) = c0;

        const auto subdiv   = loop_subdivide_one_step(V_test, F);
        const auto stencils = build_patch_stencils(subdiv.V_sub,
                                                   subdiv.F_sub);
        const Eigen::VectorXd m_vals = Abar * c0;
        REQUIRE(m_vals.size() == N + 12);

        for (int k = 0; k < 3; ++k) {
            CAPTURE(k);
            const auto& P = picks.P[static_cast<std::size_t>(k)];

            // Picked 12-vec via P_k * (Abar * c0).
            const Eigen::VectorXd c_picked = P * m_vals;
            REQUIRE(c_picked.size() == 12);

            // Direct: subdivide once geometrically and read the
            // canonical Cirak-Ortiz Fig.-9 12 DOFs of the matching
            // regular sub-face.
            const auto& stencil = stencils[static_cast<std::size_t>(
                sub_face_idx[static_cast<std::size_t>(k)])];
            const auto dofs = canonical_regular_dofs(stencil, subdiv.F_sub);

            Eigen::Matrix<double, 12, 1> c_direct;
            for (int slot = 0; slot < 12; ++slot) {
                c_direct(slot) = subdiv.V_sub(
                    dofs[static_cast<std::size_t>(slot)], 2);
            }

            INFO("c_picked = " << c_picked.transpose());
            INFO("c_direct = " << c_direct.transpose());
            for (int slot = 0; slot < 12; ++slot) {
                CAPTURE(slot);
                REQUIRE(c_picked(slot)
                        == Catch::Approx(c_direct(slot))
                            .epsilon(0).margin(1.0e-12));
            }
        }
    }
}

TEST_CASE("stam_picking_matrices: throws on N < 3",
          "[shell][loop][stam][picking][validation]")
{
    using chladni::shell::loop::stam_picking_matrices;
    REQUIRE_THROWS_AS(stam_picking_matrices(2),  std::invalid_argument);
    REQUIRE_THROWS_AS(stam_picking_matrices(0),  std::invalid_argument);
    REQUIRE_THROWS_AS(stam_picking_matrices(-1), std::invalid_argument);
}
