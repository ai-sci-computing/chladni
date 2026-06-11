/**
 * @file test_geometry.cpp
 * @brief Unit tests for chladni::shell mesh geometry and lumped mass.
 *
 * Two test fixtures:
 *
 * 1. A unit square in the xy plane, triangulated as
 *      F = {(0, 1, 2), (0, 2, 3)}
 *    with V = {(0,0,0), (1,0,0), (1,1,0), (0,1,0)}. Closed-form
 *    answers for every quantity:
 *      - face areas: (0.5, 0.5), sum 1.0
 *      - edges: 5 unique (4 boundary + 1 interior diagonal)
 *      - per-vertex masses with rho=h=1: (1/3, 1/6, 1/3, 1/6),
 *        which sum to 1.0 (the surface area).
 *
 * 2. The cylinder.obj asset: 128 vertices, 224 faces, an open shell.
 *    Geometric closed forms:
 *      - sum of face areas equals the lateral surface area of the
 *        sampled cylinder (radius 1, length 4) up to discretisation
 *        error: ~ 2 * pi * 1 * 4 = 25.133, with the 16-around polygon
 *        approximation rounding it down by ~2%.
 *      - boundary edges = 2 * 16 = 32 (the open ends of the strip).
 *      - lumped masses sum exactly to total_area * rho * h.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// 4-vertex unit square in xy, triangulated by (0,1,2) and (0,2,3).
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

TEST_CASE("face_areas: unit square triangulation gives (0.5, 0.5)",
          "[shell][geometry][areas]")
{
    const auto sq = make_unit_square();
    const auto A = chladni::shell::face_areas(sq.V, sq.F);
    REQUIRE(A.size() == 2);
    REQUIRE(A(0) == Catch::Approx(0.5).margin(1e-12));
    REQUIRE(A(1) == Catch::Approx(0.5).margin(1e-12));
    REQUIRE(A.sum() == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("face_areas: cylinder.obj sum approximates 2 pi R L within ~2%",
          "[shell][geometry][areas][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto A = chladni::shell::face_areas(mesh.V, mesh.F);
    REQUIRE(A.size() == 224);

    // 16-around polygon inscribed in unit circle has perimeter
    // 16 * 2 sin(pi / 16) ~ 6.243; analytic 2 pi ~ 6.283.
    // Total area ~ perimeter * length = 6.243 * 4 = 24.97.
    const double rim_perimeter =
        16.0 * 2.0 * std::sin(std::numbers::pi_v<double> / 16.0);
    const double expected = rim_perimeter * 4.0;
    REQUIRE(A.sum() == Catch::Approx(expected).margin(1e-6));
    REQUIRE((A.array() > 0.0).all());
}

TEST_CASE("build_edges: unit square has 5 edges (4 boundary + 1 interior)",
          "[shell][geometry][edges]")
{
    const auto sq = make_unit_square();
    const auto edges = chladni::shell::build_edges(sq.F);
    REQUIRE(edges.size() == 5);

    int n_interior = 0, n_boundary = 0;
    for (const auto& e : edges) {
        if (e.is_interior()) ++n_interior;
        else                 ++n_boundary;
    }
    REQUIRE(n_interior == 1);
    REQUIRE(n_boundary == 4);
}

TEST_CASE("build_edges: cylinder.obj has 32 boundary edges (open ends)",
          "[shell][geometry][edges][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const auto edges = chladni::shell::build_edges(mesh.F);

    // 16-around x 8-along open cylinder: total edges = 3*F/2 + boundary/2
    // arithmetic: 3F = 2*interior + boundary, so interior = (3F - boundary)/2
    int n_boundary = 0;
    for (const auto& e : edges) {
        if (e.is_boundary()) ++n_boundary;
    }
    REQUIRE(n_boundary == 32);  // 16 around * 2 ends
    // (3*224 + 32) / 2 = 352
    REQUIRE(static_cast<int>(edges.size()) == 352);
}

TEST_CASE("lumped_vertex_masses: unit square with rho=h=1",
          "[shell][geometry][mass]")
{
    const auto sq = make_unit_square();
    const auto m = chladni::shell::lumped_vertex_masses(sq.V, sq.F, 1.0, 1.0);
    REQUIRE(m.size() == 4);
    // Vertex 0 and 2 are in both triangles, vertex 1 and 3 in only one.
    REQUIRE(m(0) == Catch::Approx(1.0 / 3.0).margin(1e-12));
    REQUIRE(m(1) == Catch::Approx(1.0 / 6.0).margin(1e-12));
    REQUIRE(m(2) == Catch::Approx(1.0 / 3.0).margin(1e-12));
    REQUIRE(m(3) == Catch::Approx(1.0 / 6.0).margin(1e-12));
    REQUIRE(m.sum() == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("lumped_vertex_masses: cylinder.obj sums to total_area * rho * h",
          "[shell][geometry][mass][cylinder]")
{
    const auto mesh = chladni::mesh::load_obj(
        fs::path{CHLADNI_DATA_DIR} / "cylinder.obj");
    const double rho = 7850.0, h = 1.0e-3;
    const auto m  = chladni::shell::lumped_vertex_masses(mesh.V, mesh.F, rho, h);
    const auto A = chladni::shell::face_areas(mesh.V, mesh.F);
    REQUIRE(m.sum() == Catch::Approx(rho * h * A.sum()).margin(1e-9));
}

TEST_CASE("assemble_mass_matrix: 3n x 3n diagonal with replicated masses",
          "[shell][mass][matrix]")
{
    Eigen::VectorXd m(3);
    m << 0.5, 1.5, 2.5;

    const auto M = chladni::shell::assemble_mass_matrix(m);

    REQUIRE(M.rows() == 9);
    REQUIRE(M.cols() == 9);
    REQUIRE(M.nonZeros() == 9);

    for (Eigen::Index i = 0; i < 3; ++i) {
        for (Eigen::Index k = 0; k < 3; ++k) {
            REQUIRE(M.coeff(3 * i + k, 3 * i + k)
                    == Catch::Approx(m(i)).margin(1e-12));
        }
    }
}
