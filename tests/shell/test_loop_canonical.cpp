/**
 * @file test_loop_canonical.cpp
 * @brief Tests for canonical_regular_dofs (Cirak-Ortiz Fig. 9 ordering).
 *
 * Builds the smallest mesh that exercises the topology walk: the 12
 * control vertices of @cite stam_1999_loop_evaluation Fig. 1 placed at
 * the standard hexagonal positions in the xy-plane, with the 13
 * triangles needed to give the central triangle (vertices 4, 7, 8 in
 * 1-indexed Fig. 9 notation; rows 3, 6, 7 in 0-indexed) a closed
 * 1-ring with all-6 corner valences.
 *
 * Vertex layout (1-indexed slot k at row k-1 of V; positions chosen so
 * that the central triangle 4-7-8 points downward):
 *
 *      2     5     9                  y = sqrt(3)         row index
 *   1     4     8     12              y = sqrt(3)/2       (slots 0..3, 7, 11)
 *      3     7     11                 y = 0
 *         6     10                    y = -sqrt(3)/2
 *
 * Triangulation (13 triangles; central tri at row 0):
 *   (4,7,8) (4,8,5) (4,5,2) (4,2,1) (4,1,3) (4,3,7)    [around vertex 4]
 *   (7,3,6) (7,6,10) (7,10,11) (7,11,8)                [around vertex 7]
 *   (8,11,12) (8,12,9) (8,9,5)                         [around vertex 8]
 *
 * Test expectations:
 *  1. build_patch_stencils correctly identifies the central triangle's
 *     corner valences as (6, 6, 6) (though has_boundary=true since the
 *     outer ring vertices sit on the mesh boundary).
 *  2. canonical_regular_dofs returns [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *     for the central triangle since the mesh is constructed in slot
 *     order.
 *  3. canonical_regular_dofs throws on a triangle whose corner is on
 *     the boundary (open-fan corner — the topology walk fails).
 */

#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>
#include <numbers>

namespace {

struct Hex12Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

// Build the Stam Fig. 1 / Cirak-Ortiz Fig. 9 reference mesh.
//
// Vertices are stored in V.row(k) for slot k (= vertex k+1 in the
// 1-indexed paper convention). The central triangle is F.row(0).
Hex12Mesh make_hex12_mesh()
{
    constexpr double h = 0.8660254037844386;  // sqrt(3) / 2
    Hex12Mesh m;
    m.V.resize(12, 3);
    // slot 0 (vertex 1):  middle row, far left
    m.V.row(0) << -0.5,         h,   0.0;
    // slot 1 (vertex 2):  top row, left
    m.V.row(1) <<  0.0,    2.0 * h,  0.0;
    // slot 2 (vertex 3):  bottom row, left
    m.V.row(2) <<  0.0,    0.0,      0.0;
    // slot 3 (vertex 4):  middle row, left-center  <- corner 0
    m.V.row(3) <<  0.5,         h,   0.0;
    // slot 4 (vertex 5):  top row, center
    m.V.row(4) <<  1.0,    2.0 * h,  0.0;
    // slot 5 (vertex 6):  far bottom row, left
    m.V.row(5) <<  0.5,        -h,   0.0;
    // slot 6 (vertex 7):  bottom row, center  <- corner 1
    m.V.row(6) <<  1.0,    0.0,      0.0;
    // slot 7 (vertex 8):  middle row, right-center  <- corner 2
    m.V.row(7) <<  1.5,         h,   0.0;
    // slot 8 (vertex 9):  top row, right
    m.V.row(8) <<  2.0,    2.0 * h,  0.0;
    // slot 9 (vertex 10): far bottom row, right
    m.V.row(9) <<  1.5,        -h,   0.0;
    // slot 10 (vertex 11): bottom row, right
    m.V.row(10) << 2.0,    0.0,      0.0;
    // slot 11 (vertex 12): middle row, far right
    m.V.row(11) << 2.5,         h,   0.0;

    // 13 triangles, listed CCW when viewed from +z. Vertex indices are
    // 0-indexed (slot numbers).
    m.F.resize(13, 3);
    // Central triangle (4, 7, 8) -> slots (3, 6, 7).
    m.F.row(0)  << 3, 6, 7;
    // Around vertex 4 (slot 3), CCW after the central triangle:
    m.F.row(1)  << 3, 7, 4;   // (4, 8, 5)
    m.F.row(2)  << 3, 4, 1;   // (4, 5, 2)
    m.F.row(3)  << 3, 1, 0;   // (4, 2, 1)
    m.F.row(4)  << 3, 0, 2;   // (4, 1, 3)
    m.F.row(5)  << 3, 2, 6;   // (4, 3, 7)
    // Around vertex 7 (slot 6), the new triangles (already counted:
    // (7,8,4) = central winding, (7,4,3) = m.F.row(5) winding):
    m.F.row(6)  << 6, 2, 5;   // (7, 3, 6)
    m.F.row(7)  << 6, 5, 9;   // (7, 6, 10)
    m.F.row(8)  << 6, 9, 10;  // (7, 10, 11)
    m.F.row(9)  << 6, 10, 7;  // (7, 11, 8)
    // Around vertex 8 (slot 7), the new triangles (already counted:
    // (8,4,7) = central winding, (8,5,4) = m.F.row(1) winding,
    // (8,7,11) = m.F.row(9) winding):
    m.F.row(10) << 7, 10, 11; // (8, 11, 12)
    m.F.row(11) << 7, 11, 8;  // (8, 12, 9)
    m.F.row(12) << 7, 8,  4;  // (8, 9, 5)

    return m;
}

}  // namespace

TEST_CASE("hex12 mesh: central triangle has all-6 corner valences",
          "[shell][loop][canonical][hex12]")
{
    const auto m = make_hex12_mesh();
    const auto patches = chladni::shell::loop::build_patch_stencils(m.V, m.F);

    REQUIRE(patches.size() == 13);

    // Central triangle is F.row(0) -> patches[0]. Corners (3, 6, 7).
    const auto& p = patches[0];
    REQUIRE(p.tri_index == 0);
    REQUIRE(p.corners[0] == 3);
    REQUIRE(p.corners[1] == 6);
    REQUIRE(p.corners[2] == 7);
    REQUIRE(p.corner_valences[0] == 6);
    REQUIRE(p.corner_valences[1] == 6);
    REQUIRE(p.corner_valences[2] == 6);
    // 9 ring vertices = 12 total - 3 corners.
    REQUIRE(p.ring.size() == 9);
    // The 12-vertex mesh's outer hex sits on the boundary, so
    // has_boundary is true even though the 3 corners themselves are
    // interior (closed fans). is_regular() therefore returns false
    // (the boundary basis would be needed for K assembly here), but
    // canonical_regular_dofs still applies because the corner walks
    // are closed.
    REQUIRE(p.has_boundary);
    REQUIRE_FALSE(p.is_regular());
}

TEST_CASE("canonical_regular_dofs: hex12 central triangle yields slot identity",
          "[shell][loop][canonical][hex12]")
{
    const auto m = make_hex12_mesh();
    const auto patches = chladni::shell::loop::build_patch_stencils(m.V, m.F);
    const auto dofs =
        chladni::shell::loop::canonical_regular_dofs(patches[0], m.F);

    // Mesh was constructed with V.row(k) at slot k, so the canonical
    // ordering must be the identity 0, 1, ..., 11.
    for (int k = 0; k < 12; ++k) {
        CAPTURE(k);
        REQUIRE(dofs[static_cast<std::size_t>(k)] == k);
    }
}

TEST_CASE("canonical_regular_dofs: throws on a non-regular-valence corner",
          "[shell][loop][canonical][error]")
{
    const auto m = make_hex12_mesh();
    const auto patches = chladni::shell::loop::build_patch_stencils(m.V, m.F);

    // patches[1] is (4, 8, 5) = slots (3, 7, 4). Corner valences:
    //   slot 3 (vertex 4) = 6
    //   slot 7 (vertex 8) = 6
    //   slot 4 (vertex 5) = 4 (boundary vertex, valence < 6)
    // canonical_regular_dofs must reject this stencil.
    REQUIRE_THROWS_AS(
        chladni::shell::loop::canonical_regular_dofs(patches[1], m.F),
        std::invalid_argument);
}
