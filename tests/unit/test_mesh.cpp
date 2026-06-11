/**
 * @file test_mesh.cpp
 * @brief Unit tests for the OBJ loader.
 *
 * Pins @ref chladni::mesh::load_obj to the topology of `models/cylinder.obj`,
 * a 16-around x 8-along open cylindrical strip used as the canonical
 * drum-shell test mesh.
 *
 *   header (from the file): "Cylinder: 16 around, 8 along, radius=1, height=4"
 *   expected V.rows() == 128
 *   expected F.rows() == 224  (16 panels * 7 axial gaps * 2 tris per quad)
 *   first vertex (line 3 of file) == (1, 0, 0)
 */

#include <chladni/mesh.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

namespace {

namespace fs = std::filesystem;

/// Resolve `models/cylinder.obj` from the source tree. The path is fed in
/// at compile time so the test works regardless of `ctest`'s cwd.
fs::path test_mesh_path(const char* relative)
{
    return fs::path{CHLADNI_DATA_DIR} / relative;
}

}  // namespace

TEST_CASE("load_obj: cylinder.obj has the expected topology",
          "[mesh][obj][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(test_mesh_path("cylinder.obj"));

    SECTION("vertex count matches the file header (16 x 8 grid)") {
        REQUIRE(mesh.num_vertices() == 128);
    }

    SECTION("face count matches the consistent-diagonal triangulation") {
        // 16 panels around * 7 axial gaps * 2 triangles per quad.
        REQUIRE(mesh.num_faces() == 224);
    }

    SECTION("first vertex is (1, 0, 0) on the unit-circle ring") {
        REQUIRE(mesh.V.rows() >= 1);
        REQUIRE(mesh.V(0, 0) == 1.0);
        REQUIRE(mesh.V(0, 1) == 0.0);
        REQUIRE(mesh.V(0, 2) == 0.0);
    }

    SECTION("all face indices lie in [0, num_vertices)") {
        for (Eigen::Index i = 0; i < mesh.F.rows(); ++i) {
            for (Eigen::Index j = 0; j < 3; ++j) {
                const auto idx = mesh.F(i, j);
                REQUIRE(idx >= 0);
                REQUIRE(idx < mesh.num_vertices());
            }
        }
    }
}

TEST_CASE("load_obj: missing file throws", "[mesh][obj][validation]")
{
    REQUIRE_THROWS_AS(
        chladni::mesh::load_obj(test_mesh_path("does_not_exist.obj")),
        std::runtime_error);
}
