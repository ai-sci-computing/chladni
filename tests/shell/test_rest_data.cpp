/**
 * @file test_rest_data.cpp
 * @brief Unit tests for chladni::shell::compute_edge_rest_data.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Core>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct UnitSquare {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

UnitSquare make_unit_square()
{
    UnitSquare s;
    s.V.resize(4, 3);
    s.V << 0.0, 0.0, 0.0,
           1.0, 0.0, 0.0,
           1.0, 1.0, 0.0,
           0.0, 1.0, 0.0;
    s.F.resize(2, 3);
    s.F << 0, 1, 2,
           0, 2, 3;
    return s;
}

}  // namespace

TEST_CASE("build_edges: rejects vertex-non-manifold input (two flaps at one point)",
          "[shell][rest_data][manifold]")
{
    // Two triangles share only vertex 0 (no shared edge), so vertex 0's two
    // incident faces form two disconnected fans — edge-manifold but pinched.
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         0, 3, 4;

    REQUIRE_THROWS_WITH(
        chladni::shell::build_edges(F),
        Catch::Matchers::ContainsSubstring("non-manifold")
            && Catch::Matchers::ContainsSubstring("vertex 0"));

    // A clean single fan around vertex 0 (the unit square) must NOT throw.
    const auto sq = make_unit_square();
    REQUIRE_NOTHROW(chladni::shell::build_edges(sq.F));
}

TEST_CASE("build_edges: rejects disconnected input (two disjoint triangles)",
          "[shell][rest_data][manifold]")
{
    // Two triangles sharing no vertex — each is vertex-manifold (every
    // vertex bounds a single fan) but together they span two connected
    // components. compute_shell_modes projects out only one 6-DOF rigid
    // subspace, so the second component's rigid modes would slip through
    // the rigid filter as spurious ~0 Hz physical modes; build_edges must
    // reject the input up front.
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 2,
         3, 4, 5;

    REQUIRE_THROWS_WITH(
        chladni::shell::build_edges(F),
        Catch::Matchers::ContainsSubstring("disconnected")
            && Catch::Matchers::ContainsSubstring("connected components"));

    // A single connected component (the unit square) must NOT throw.
    const auto sq = make_unit_square();
    REQUIRE_NOTHROW(chladni::shell::build_edges(sq.F));
}

TEST_CASE("edge_rest_data: unit square — flat rest, length and h_e match closed forms",
          "[shell][rest_data]")
{
    const auto sq = make_unit_square();
    const auto edges = chladni::shell::build_edges(sq.F);
    const auto rd = chladni::shell::compute_edge_rest_data(sq.V, sq.F, edges);

    REQUIRE(rd.size() == edges.size());

    // Edges are sorted by (v0, v1): (0,1), (0,2), (0,3), (1,2), (2,3).
    const auto& e_diag = edges[1];   // (0,2) is the only interior edge
    REQUIRE(e_diag.v0 == 0);
    REQUIRE(e_diag.v1 == 2);
    REQUIRE(e_diag.is_interior());

    const auto& d_diag = rd[1];

    // The diagonal of a unit square has length sqrt(2).
    REQUIRE(d_diag.length == Catch::Approx(std::sqrt(2.0)).margin(1e-12));

    // Both faces are coplanar, so rest dihedral is zero.
    REQUIRE(d_diag.dihedral == Catch::Approx(0.0).margin(1e-10));

    // Each triangle has area 1/2, base = sqrt(2). Height = 2A/base = sqrt(2)/2.
    // h_e = (1/3) * mean of incident face heights = (1/3) * sqrt(2)/2.
    REQUIRE(d_diag.h_e == Catch::Approx(std::sqrt(2.0) / 6.0).margin(1e-12));

    // Wings: face (0,1,2) has third vertex 1; face (0,2,3) has third vertex 3.
    // Which one is left vs. right depends on which face has the directed edge
    // 0->2; in face (0,1,2) the edges are 0-1, 1-2, 2-0, so 0->2 is reversed
    // (we have 2->0). In face (0,2,3) the edges are 0-2, 2-3, 3-0, so 0->2 is
    // present. So face_left = 1 (face index of (0,2,3)), c_left = 3.
    REQUIRE(d_diag.c_left == 3);
    REQUIRE(d_diag.c_right == 1);

    // Edge unit vector v0->v1 is the diagonal direction.
    const Eigen::Vector3d expected_hat(1.0 / std::sqrt(2.0),
                                       1.0 / std::sqrt(2.0),
                                       0.0);
    REQUIRE((d_diag.hat - expected_hat).norm() < 1e-12);

    // Boundary edges have sentinels.
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].is_boundary()) {
            REQUIRE(rd[i].dihedral == 0.0);
            REQUIRE((rd[i].c_left == -1 || rd[i].c_right == -1));
        }
    }
}

TEST_CASE("edge_rest_data: cylinder.obj — rest dihedrals positive, bounded by panel angle",
          "[shell][rest_data][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto edges = chladni::shell::build_edges(mesh.F);
    const auto rd = chladni::shell::compute_edge_rest_data(mesh.V, mesh.F, edges);
    REQUIRE(rd.size() == edges.size());

    // For a 16-around polygon, the angle between adjacent vertical panels is
    // 2 pi / 16. Three classes of interior edge:
    //   - Vertical (between adjacent panels around the cylinder): 16 * 7 = 112,
    //     dihedral approx = panel_angle.
    //   - Quad-diagonal (within a flat panel): 16 * 7 = 112, dihedral = 0.
    //   - Horizontal interior (between stacked bands, no axial curvature):
    //     16 * 6 = 96, dihedral = 0.
    // Total: 320 interior edges (= 352 - 32 boundary).
    const double panel_angle = 2.0 * std::numbers::pi_v<double> / 16.0;

    int n_curved = 0, n_zero = 0;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (!edges[i].is_interior()) continue;
        const double t = std::abs(rd[i].dihedral);
        if (t > 0.5 * panel_angle) {
            REQUIRE(t < 1.5 * panel_angle);
            ++n_curved;
        } else {
            REQUIRE(t < 1e-6);
            ++n_zero;
        }
    }
    REQUIRE(n_curved == 112);  // vertical panel-to-panel edges
    REQUIRE(n_zero   == 208);  // 112 quad-diagonals + 96 horizontal interior
}
