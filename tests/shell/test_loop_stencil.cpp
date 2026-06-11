/**
 * @file test_loop_stencil.cpp
 * @brief Unit tests for the Loop subdivision patch-stencil enumerator.
 *
 * The Loop subdivision shell formulation (Cirak-Ortiz-Schroder 2000)
 * builds per-triangle stiffness from a 1-ring patch around each
 * triangle: 12 control vertices for "regular" patches (all 3 corners
 * have vertex valence = 6), @f$N+6@f$ control vertices for "irregular"
 * patches with one corner of valence @f$N \neq 6@f$. Patches with any
 * boundary vertex are flagged for the Schweitzer/Hoppe phantom-vertex
 * treatment that lands in L.5.
 *
 * Two fixtures:
 *
 * 1. **Unit square** (4 vertices, 2 triangles, 5 edges). Hand-verifiable
 *    valences (3, 2, 3, 2). Both triangles are boundary patches with
 *    irregular corner valences. The ring of each triangle is exactly
 *    the opposite vertex of the diagonal.
 *
 * 2. **cylinder.obj** (128 vertices, 224 faces, an open shell).
 *    Diagnostic test: the open ends contribute boundary patches; the
 *    interior of the strip should contain regular triangles (every
 *    interior vertex has valence 6 in the 16-around / 8-along
 *    triangulation).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <filesystem>

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

TEST_CASE("vertex_valences: unit square gives (3, 2, 3, 2)",
          "[shell][loop][stencil][valence]")
{
    const auto sq    = make_unit_square();
    const auto edges = chladni::shell::build_edges(sq.F);
    const auto val   = chladni::shell::loop::vertex_valences(
        static_cast<Eigen::Index>(sq.V.rows()), edges);

    REQUIRE(val.size() == 4);
    REQUIRE(val[0] == 3);
    REQUIRE(val[1] == 2);
    REQUIRE(val[2] == 3);
    REQUIRE(val[3] == 2);
}

TEST_CASE("build_patch_stencils: unit square has 2 boundary irregular patches",
          "[shell][loop][stencil][unit_square]")
{
    const auto sq      = make_unit_square();
    const auto patches = chladni::shell::loop::build_patch_stencils(sq.V, sq.F);

    REQUIRE(patches.size() == 2);

    // Triangle 0 = (0, 1, 2), corner valences (3, 2, 3); ring = {3}.
    REQUIRE(patches[0].tri_index == 0);
    REQUIRE(patches[0].corners[0] == 0);
    REQUIRE(patches[0].corners[1] == 1);
    REQUIRE(patches[0].corners[2] == 2);
    REQUIRE(patches[0].corner_valences[0] == 3);
    REQUIRE(patches[0].corner_valences[1] == 2);
    REQUIRE(patches[0].corner_valences[2] == 3);
    REQUIRE(patches[0].ring.size() == 1);
    REQUIRE(patches[0].ring[0] == 3);
    REQUIRE(patches[0].has_boundary);
    REQUIRE_FALSE(patches[0].is_regular());
    REQUIRE(patches[0].n_dofs() == 4);
    REQUIRE(patches[0].max_corner_valence() == 3);

    // Triangle 1 = (0, 2, 3), corner valences (3, 3, 2); ring = {1}.
    REQUIRE(patches[1].tri_index == 1);
    REQUIRE(patches[1].corners[0] == 0);
    REQUIRE(patches[1].corners[1] == 2);
    REQUIRE(patches[1].corners[2] == 3);
    REQUIRE(patches[1].corner_valences[0] == 3);
    REQUIRE(patches[1].corner_valences[1] == 3);
    REQUIRE(patches[1].corner_valences[2] == 2);
    REQUIRE(patches[1].ring.size() == 1);
    REQUIRE(patches[1].ring[0] == 1);
    REQUIRE(patches[1].has_boundary);
    REQUIRE_FALSE(patches[1].is_regular());
    REQUIRE(patches[1].n_dofs() == 4);
    REQUIRE(patches[1].max_corner_valence() == 3);
}

TEST_CASE("build_patch_stencils: cylinder.obj has interior regular patches",
          "[shell][loop][stencil][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto patches = chladni::shell::loop::build_patch_stencils(
        mesh.V, mesh.F);

    REQUIRE(patches.size() == 224);

    // Every triangle is either boundary or has all-6 corner valences.
    // Cylinder.obj is a 16-around x 8-along strip; interior vertices
    // are valence 6 in the triangulation.
    int n_regular = 0;
    int n_boundary = 0;
    int n_irregular_interior = 0;
    for (const auto& p : patches) {
        if (p.is_regular()) {
            ++n_regular;
            REQUIRE(p.n_dofs() == 12);  // canonical regular patch DOF count
        } else if (p.has_boundary) {
            ++n_boundary;
        } else {
            ++n_irregular_interior;
        }
    }

    // The 16 boundary-edge rings at z=0 and z=L make ~32 boundary
    // triangles per end whose 1-ring touches a boundary vertex. The
    // bulk are regular interior. We assert "at least one of each
    // category exists" without pinning the exact count.
    REQUIRE(n_regular > 0);
    REQUIRE(n_boundary > 0);
    REQUIRE(n_irregular_interior == 0);  // no extraordinary vertices on this mesh
    REQUIRE(n_regular + n_boundary == 224);
}
