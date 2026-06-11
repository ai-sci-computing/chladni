/**
 * @file test_loop_gather_stam_dofs.cpp
 * @brief S.7b — gather_stam_patch_dofs round-trip test on the canonical
 *        Stam patch.
 *
 * Pins @ref chladni::shell::loop::gather_stam_patch_dofs by running it
 * on the canonical irregular patch built by @c build_irregular_patch
 * (mirrored here from loop_stam.cpp / test_loop_stam_element_K.cpp) and
 * checking the returned DOF list is exactly the identity {0, 1, ..., K-1}
 * — i.e. the gather routine recovers Stam's canonical vertex ordering
 * verbatim. This is the topological inverse of build_irregular_patch:
 * the patch was constructed with vertex labels matching the canonical
 * Stam slots, so gather must agree.
 *
 * Also exercises the error paths:
 *  - regular patch (all valence 6) — throws with "use canonical_regular_dofs".
 *  - multiple extraordinary corners — throws with "more than one corner".
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// Mirror of build_irregular_patch in loop_stam.cpp.
void build_irregular_patch(int N,
                           Eigen::MatrixXd& V,
                           Eigen::MatrixXi& F)
{
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

TEST_CASE("gather_stam_patch_dofs: identity round-trip on canonical patch",
          "[shell][loop][stam][gather]")
{
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::gather_stam_patch_dofs;

    // Sweep valences spanning N=3 (Jordan special case) through 8.
    for (int N : {3, 4, 5, 6, 7, 8}) {
        if (N == 6) continue;  // not extraordinary; covered separately
        SECTION(("N=" + std::to_string(N)).c_str()) {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            build_irregular_patch(N, V, F);

            const auto patches = build_patch_stencils(V, F);
            REQUIRE(patches.size() == static_cast<std::size_t>(F.rows()));

            // Central face F.row(0) = (0, 1, N): extraordinary vertex
            // at slot k_ev = 0.
            const auto& central = patches[0];
            REQUIRE(central.tri_index == 0);
            REQUIRE(central.corner_valences[0] == N);
            REQUIRE(central.corner_valences[1] == 6);
            REQUIRE(central.corner_valences[2] == 6);

            const auto dofs = gather_stam_patch_dofs(central, F);
            const int K = N + 6;
            REQUIRE(static_cast<int>(dofs.size()) == K);

            // The canonical patch was constructed with vertex labels
            // matching Stam's canonical slot ordering, so gather must
            // return identity.
            for (int i = 0; i < K; ++i) {
                REQUIRE(dofs[static_cast<std::size_t>(i)] == i);
            }
        }
    }
}

TEST_CASE("gather_stam_patch_dofs: round-trip when the extraordinary "
          "corner is at slot 1 or 2",
          "[shell][loop][stam][gather]")
{
    // Re-roll the central face's corner order. Stam-canonical has the
    // extraordinary corner at F(0, 0); permute to F(0, 0) = 1 ->
    // (1, N, 0) and (N, 0, 1). gather_stam_patch_dofs must auto-detect
    // the extraordinary slot (whichever corner has valence != 6) and
    // produce the same set of DOFs (re-rolled accordingly).
    using chladni::shell::loop::build_patch_stencils;
    using chladni::shell::loop::gather_stam_patch_dofs;

    const int N = 5;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    build_irregular_patch(N, V, F);

    // Identity gather first.
    const auto patches_ref = build_patch_stencils(V, F);
    const auto dofs_ref    = gather_stam_patch_dofs(patches_ref[0], F);

    for (int shift : {1, 2}) {
        SECTION(("shift=" + std::to_string(shift)).c_str()) {
            Eigen::MatrixXi F_shift = F;
            // Rotate F.row(0) CCW by `shift` positions.
            std::array<int, 3> row0 = {F(0, 0), F(0, 1), F(0, 2)};
            for (int k = 0; k < 3; ++k) {
                F_shift(0, k) = row0[
                    static_cast<std::size_t>((k + shift) % 3)];
            }

            const auto patches = build_patch_stencils(V, F_shift);
            const auto& central = patches[0];

            // Extraordinary corner now at slot (3 - shift) % 3 because
            // we rotated forward; the valence-N vertex (vertex 0)
            // ends up at corner index (-shift mod 3).
            const int expected_kev = (3 - shift) % 3;
            REQUIRE(central.corner_valences[
                static_cast<std::size_t>(expected_kev)] == N);

            const auto dofs = gather_stam_patch_dofs(central, F_shift);
            // Same K and same canonical DOFs.
            REQUIRE(dofs.size() == dofs_ref.size());
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                REQUIRE(dofs[i] == dofs_ref[i]);
            }
        }
    }
}

TEST_CASE("gather_stam_patch_dofs: throws when all corners are valence 6",
          "[shell][loop][stam][gather][validation]")
{
    using chladni::shell::loop::PatchStencil;
    using chladni::shell::loop::gather_stam_patch_dofs;

    PatchStencil p;
    p.tri_index       = 0;
    p.corners         = {0, 1, 2};
    p.corner_valences = {6, 6, 6};
    p.has_boundary    = false;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;
    REQUIRE_THROWS_AS(gather_stam_patch_dofs(p, F), std::invalid_argument);
}

TEST_CASE("gather_stam_patch_dofs: throws on multiple extraordinary corners",
          "[shell][loop][stam][gather][validation]")
{
    using chladni::shell::loop::PatchStencil;
    using chladni::shell::loop::gather_stam_patch_dofs;

    PatchStencil p;
    p.tri_index       = 0;
    p.corners         = {0, 1, 2};
    p.corner_valences = {5, 5, 6};
    p.has_boundary    = false;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;
    REQUIRE_THROWS_AS(gather_stam_patch_dofs(p, F), std::invalid_argument);
}
