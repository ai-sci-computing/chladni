/**
 * @file test_loop_subdivide.cpp
 * @brief Unit tests for one-step Loop subdivision (L.3.4a).
 *
 * Pins down combinatorics, the four Loop masks, and the constraint
 * matrix S that captures the subdivision step as a linear operator on
 * stacked DOF displacements.
 *
 * Fixtures:
 *  - **Octahedron** (6 vertices, 8 faces, 12 interior edges; every
 *    vertex has valence 4, i.e. *extraordinary* for Loop). Used to
 *    exercise the closed-mesh path with no boundary.
 *  - **Single triangle** (3 vertices, 1 face, 3 boundary edges). Used
 *    to verify the boundary odd rule (midpoint rule) and even rule.
 *  - **Two-triangle quad** (4 vertices, 2 faces, 5 edges with 1
 *    interior). Used to verify the interior odd rule on the diagonal
 *    edge in isolation.
 *  - **Flat hex-12 patch** (the test_loop_canonical fixture). Used to
 *    verify Loop's β formula at a regular interior valence-6 vertex
 *    and to verify that subdivision preserves planarity exactly on a
 *    flat mesh.
 */

#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <cmath>
#include <numbers>

using Catch::Matchers::WithinAbs;

namespace {

struct Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

// Octahedron: 6 vertices on the +x, -x, +y, -y, +z, -z axes; 8
// equilateral-ish faces. Every vertex has valence 4.
Mesh make_octahedron()
{
    Mesh m;
    m.V.resize(6, 3);
    m.V <<  1,  0,  0,
           -1,  0,  0,
            0,  1,  0,
            0, -1,  0,
            0,  0,  1,
            0,  0, -1;
    m.F.resize(8, 3);
    // Top hemisphere (vertex 4 = +z), CCW seen from +z:
    m.F << 0, 2, 4,
           2, 1, 4,
           1, 3, 4,
           3, 0, 4,
    // Bottom hemisphere (vertex 5 = -z), CCW seen from -z (opposite winding):
           2, 0, 5,
           1, 2, 5,
           3, 1, 5,
           0, 3, 5;
    return m;
}

Mesh make_single_triangle()
{
    Mesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0;
    m.F.resize(1, 3);
    m.F << 0, 1, 2;
    return m;
}

// 4 vertices in a unit square, two CCW triangles meeting at the
// (0)-(2) diagonal. The diagonal is the only interior edge.
Mesh make_two_triangle_quad()
{
    Mesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0,
           1, 0, 0,
           1, 1, 0,
           0, 1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2,
           0, 2, 3;
    return m;
}

// Flat hex-12 patch: same as test_loop_canonical's make_hex12_mesh.
// Slot 3 is the central interior valence-6 vertex; the other 11 sit
// on the mesh boundary.
Mesh make_flat_hex12()
{
    constexpr double h = 0.8660254037844386;
    Mesh m;
    m.V.resize(12, 3);
    m.V.row(0)  << -0.5,         h,   0.0;
    m.V.row(1)  <<  0.0,    2.0 * h,  0.0;
    m.V.row(2)  <<  0.0,    0.0,      0.0;
    m.V.row(3)  <<  0.5,         h,   0.0;
    m.V.row(4)  <<  1.0,    2.0 * h,  0.0;
    m.V.row(5)  <<  0.5,        -h,   0.0;
    m.V.row(6)  <<  1.0,    0.0,      0.0;
    m.V.row(7)  <<  1.5,         h,   0.0;
    m.V.row(8)  <<  2.0,    2.0 * h,  0.0;
    m.V.row(9)  <<  1.5,        -h,   0.0;
    m.V.row(10) <<  2.0,    0.0,      0.0;
    m.V.row(11) <<  2.5,         h,   0.0;

    m.F.resize(13, 3);
    m.F.row(0)  << 3, 6, 7;
    m.F.row(1)  << 3, 7, 4;
    m.F.row(2)  << 3, 4, 1;
    m.F.row(3)  << 3, 1, 0;
    m.F.row(4)  << 3, 0, 2;
    m.F.row(5)  << 3, 2, 6;
    m.F.row(6)  << 6, 2, 5;
    m.F.row(7)  << 6, 5, 9;
    m.F.row(8)  << 6, 9, 10;
    m.F.row(9)  << 6, 10, 7;
    m.F.row(10) << 7, 10, 11;
    m.F.row(11) << 7, 11, 8;
    m.F.row(12) << 7, 8,  4;
    return m;
}

}  // namespace

TEST_CASE("loop_subdivide_one_step: octahedron combinatorics",
          "[shell][loop][subdivide][closed]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    REQUIRE(sub.n_real == 6);
    REQUIRE(sub.n_real_faces == 8);
    REQUIRE(sub.n_edge_midpoints == 12);
    REQUIRE(sub.V_sub.rows() == 6 + 12);
    REQUIRE(sub.F_sub.rows() == 4 * 8);

    // S has shape 3*(6+12) x 3*6 = 54 x 18.
    REQUIRE(sub.S.rows() == 54);
    REQUIRE(sub.S.cols() == 18);
}

TEST_CASE("loop_subdivide_one_step: octahedron interior even-rule weights",
          "[shell][loop][subdivide][even]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // Every octahedron vertex is interior with valence 4. Loop's β_4:
    //   c4 = 3/8 + 1/4 * cos(π/2) = 3/8
    //   β_4 = (1/4) * (5/8 - (3/8)^2) = (1/4) * (5/8 - 9/64)
    //       = (1/4) * (40/64 - 9/64) = (1/4) * (31/64) = 31/256
    const double beta4 = (1.0 / 4.0) * (5.0 / 8.0 - (3.0 / 8.0) * (3.0 / 8.0));
    REQUIRE_THAT(beta4, WithinAbs(31.0 / 256.0, 1e-15));

    // Vertex 4 = +z (1, 0, 0, 1)... actually m.V.row(4) = (0, 0, 1).
    // Its neighbours are 0, 1, 2, 3 (all 4 on the equator). Sum = (0,0,0).
    // Smoothed position:
    //   v' = (1 - 4*β_4) * (0,0,1) + β_4 * (0,0,0)
    //      = (1 - 4*31/256) * (0,0,1)
    //      = (1 - 124/256) * (0,0,1)
    //      = (132/256) * (0,0,1) = (33/64) * (0,0,1)
    const double w_centre = 1.0 - 4.0 * beta4;  // 33/64
    REQUIRE_THAT(w_centre, WithinAbs(33.0 / 64.0, 1e-15));

    REQUIRE_THAT(sub.V_sub(4, 0), WithinAbs(0.0,           1e-15));
    REQUIRE_THAT(sub.V_sub(4, 1), WithinAbs(0.0,           1e-15));
    REQUIRE_THAT(sub.V_sub(4, 2), WithinAbs(33.0 / 64.0,   1e-15));

    // Vertex 0 = +x (1, 0, 0). Its neighbours are 2, 3, 4, 5
    // (i.e. ±y and ±z, sum = 0). Smoothed:
    //   v' = (33/64) * (1,0,0) + β_4 * (0,0,0) = (33/64, 0, 0).
    REQUIRE_THAT(sub.V_sub(0, 0), WithinAbs(33.0 / 64.0,   1e-15));
    REQUIRE_THAT(sub.V_sub(0, 1), WithinAbs(0.0,           1e-15));
    REQUIRE_THAT(sub.V_sub(0, 2), WithinAbs(0.0,           1e-15));
}

TEST_CASE("loop_subdivide_one_step: octahedron interior odd-rule weights",
          "[shell][loop][subdivide][odd]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // build_edges sorts edges lexicographically by (v0, v1) with v0<v1.
    // Edge 0 is (0, 2). Adjacent faces: F.row(0) = (0,2,4) → opposite
    // vertex 4; F.row(4) = (2,0,5) → opposite vertex 5.
    // Loop interior odd rule:
    //   p = 3/8 (v0 + v2) + 1/8 (v4 + v5)
    //     = 3/8 ((1,0,0) + (0,1,0)) + 1/8 ((0,0,1) + (0,0,-1))
    //     = (3/8, 3/8, 0)
    REQUIRE(sub.n_real == 6);
    const Eigen::Index e0_row = 6 + 0;
    REQUIRE_THAT(sub.V_sub(e0_row, 0), WithinAbs(0.375, 1e-15));
    REQUIRE_THAT(sub.V_sub(e0_row, 1), WithinAbs(0.375, 1e-15));
    REQUIRE_THAT(sub.V_sub(e0_row, 2), WithinAbs(0.0,   1e-15));
}

TEST_CASE("loop_subdivide_one_step: single-triangle boundary even/odd rules",
          "[shell][loop][subdivide][boundary]")
{
    const auto m = make_single_triangle();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    REQUIRE(sub.n_real == 3);
    REQUIRE(sub.n_real_faces == 1);
    REQUIRE(sub.n_edge_midpoints == 3);
    REQUIRE(sub.V_sub.rows() == 6);
    REQUIRE(sub.F_sub.rows() == 4);

    // Boundary even rule for vertex 0 = (0,0,0). Boundary neighbours
    // are 1 and 2. Smoothed:
    //   v' = 3/4 * (0,0,0) + 1/8 * ((1,0,0) + (0,1,0)) = (1/8, 1/8, 0)
    REQUIRE_THAT(sub.V_sub(0, 0), WithinAbs(0.125, 1e-15));
    REQUIRE_THAT(sub.V_sub(0, 1), WithinAbs(0.125, 1e-15));
    REQUIRE_THAT(sub.V_sub(0, 2), WithinAbs(0.0,   1e-15));

    // Boundary odd rule on edge 0 = (0,1):
    //   p = 1/2 * ((0,0,0) + (1,0,0)) = (0.5, 0, 0)
    REQUIRE_THAT(sub.V_sub(3, 0), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(sub.V_sub(3, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(sub.V_sub(3, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("loop_subdivide_one_step: two-triangle quad — diagonal is interior",
          "[shell][loop][subdivide][odd]")
{
    const auto m = make_two_triangle_quad();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    REQUIRE(sub.n_real == 4);
    REQUIRE(sub.n_real_faces == 2);
    REQUIRE(sub.n_edge_midpoints == 5);

    // The diagonal edge (0, 2) is the only interior edge. Its third
    // vertices come from F.row(0) = (0,1,2) → vertex 1 and from
    // F.row(1) = (0,2,3) → vertex 3. Loop interior odd rule:
    //   p = 3/8 ((0,0,0) + (1,1,0)) + 1/8 ((1,0,0) + (0,1,0))
    //     = (3/8, 3/8, 0) + (1/8, 1/8, 0)
    //     = (1/2, 1/2, 0)
    // Find which edge index has (v0, v1) = (0, 2).
    const auto edges = chladni::shell::build_edges(m.F);
    Eigen::Index diag_edge = -1;
    for (Eigen::Index e = 0; e < static_cast<Eigen::Index>(edges.size()); ++e) {
        if (edges[static_cast<std::size_t>(e)].v0 == 0
         && edges[static_cast<std::size_t>(e)].v1 == 2)
        {
            diag_edge = e;
            break;
        }
    }
    REQUIRE(diag_edge != -1);
    const Eigen::Index row = 4 + diag_edge;
    REQUIRE_THAT(sub.V_sub(row, 0), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(sub.V_sub(row, 1), WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(sub.V_sub(row, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("loop_subdivide_one_step: hex12 central valence-6 β = 1/16",
          "[shell][loop][subdivide][even]")
{
    const auto m = make_flat_hex12();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // Slot 3 (= the only interior vertex) has valence 6 in this mesh.
    // Loop's β_6:
    //   c6 = 3/8 + 1/4 * cos(π/3) = 3/8 + 1/4 * 1/2 = 3/8 + 1/8 = 1/2
    //   β_6 = (1/6) * (5/8 - (1/2)^2) = (1/6) * (5/8 - 1/4)
    //       = (1/6) * (3/8) = 1/16
    // Sum of slot 3's 6 neighbours (slots 0, 1, 2, 4, 6, 7) on the
    // flat hex grid is exactly 6 * (slot 3's position) by symmetry,
    // so the smoothing rule reproduces the centre exactly:
    //   v' = (1 - 6*1/16) * v + 1/16 * 6*v = v
    // i.e. linear-function preservation.
    REQUIRE_THAT(sub.V_sub(3, 0), WithinAbs(m.V(3, 0), 1e-14));
    REQUIRE_THAT(sub.V_sub(3, 1), WithinAbs(m.V(3, 1), 1e-14));
    REQUIRE_THAT(sub.V_sub(3, 2), WithinAbs(m.V(3, 2), 1e-14));
}

TEST_CASE("loop_subdivide_one_step: flat-mesh planarity is preserved",
          "[shell][loop][subdivide][flat]")
{
    const auto m = make_flat_hex12();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // Loop subdivision is linear in the input positions; if every
    // input lies in the plane z=0 then every subdivided vertex must
    // too, regardless of which mask (interior/boundary, even/odd)
    // applies.
    for (Eigen::Index k = 0; k < sub.V_sub.rows(); ++k) {
        CAPTURE(k);
        REQUIRE_THAT(sub.V_sub(k, 2), WithinAbs(0.0, 1e-14));
    }
}

TEST_CASE("loop_subdivide_one_step: F_sub winding is consistent with parent",
          "[shell][loop][subdivide][topology]")
{
    const auto m = make_flat_hex12();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // For every parent face f, the 4 sub-triangles should have signed
    // 2D area in the (x, y) plane with the same sign as the parent
    // (positive since the input faces are CCW seen from +z).
    for (Eigen::Index f = 0; f < sub.n_real_faces; ++f) {
        const auto& V0 = m.V.row(m.F(f, 0));
        const auto& V1 = m.V.row(m.F(f, 1));
        const auto& V2 = m.V.row(m.F(f, 2));
        const double parent_signed = (V1(0) - V0(0)) * (V2(1) - V0(1))
                                   - (V1(1) - V0(1)) * (V2(0) - V0(0));
        CAPTURE(f, parent_signed);
        REQUIRE(parent_signed > 0.0);
        for (int s = 0; s < 4; ++s) {
            const Eigen::Index sf = 4 * f + s;
            const auto& W0 = sub.V_sub.row(sub.F_sub(sf, 0));
            const auto& W1 = sub.V_sub.row(sub.F_sub(sf, 1));
            const auto& W2 = sub.V_sub.row(sub.F_sub(sf, 2));
            const double sub_signed = (W1(0) - W0(0)) * (W2(1) - W0(1))
                                    - (W1(1) - W0(1)) * (W2(0) - W0(0));
            CAPTURE(s, sub_signed);
            REQUIRE(sub_signed > 0.0);
        }
    }
}

TEST_CASE("loop_subdivide_one_step: S reproduces V_sub on stacked DOFs",
          "[shell][loop][subdivide][constraint]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // Stack V into a length-3*n_real vector with layout
    // [V(0,0), V(0,1), V(0,2), V(1,0), V(1,1), V(1,2), ...].
    Eigen::VectorXd v_flat(3 * m.V.rows());
    for (Eigen::Index i = 0; i < m.V.rows(); ++i) {
        v_flat(3 * i + 0) = m.V(i, 0);
        v_flat(3 * i + 1) = m.V(i, 1);
        v_flat(3 * i + 2) = m.V(i, 2);
    }
    const Eigen::VectorXd vsub_flat = sub.S * v_flat;

    REQUIRE(vsub_flat.size() == 3 * sub.V_sub.rows());
    for (Eigen::Index i = 0; i < sub.V_sub.rows(); ++i) {
        CAPTURE(i);
        for (int d = 0; d < 3; ++d) {
            CAPTURE(d);
            REQUIRE_THAT(vsub_flat(3 * i + d),
                         WithinAbs(sub.V_sub(i, d), 1e-14));
        }
    }
}

TEST_CASE("loop_subdivide_n_times: n_passes = 0 is the identity",
          "[shell][loop][subdivide][multipass]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_n_times(m.V, m.F, 0);
    REQUIRE(sub.V_sub.rows() == m.V.rows());
    REQUIRE(sub.F_sub.rows() == m.F.rows());
    REQUIRE(sub.n_real == m.V.rows());
    REQUIRE(sub.n_real_faces == m.F.rows());
    REQUIRE(sub.n_edge_midpoints == 0);

    // V_sub == V exactly.
    REQUIRE(sub.V_sub.isApprox(m.V, 1e-15));
    // F_sub == F exactly.
    REQUIRE(sub.F_sub.isApprox(m.F));
    // S is identity.
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(3 * m.V.rows(), 1, 99);
    Eigen::VectorXd Sv = sub.S * v;
    REQUIRE(Sv.isApprox(v, 1e-15));
}

TEST_CASE("loop_subdivide_n_times: n_passes = 1 matches loop_subdivide_one_step",
          "[shell][loop][subdivide][multipass]")
{
    const auto m = make_octahedron();
    const auto one  = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);
    const auto many = chladni::shell::loop::loop_subdivide_n_times(m.V, m.F, 1);

    REQUIRE(many.V_sub.rows() == one.V_sub.rows());
    REQUIRE(many.F_sub.rows() == one.F_sub.rows());
    REQUIRE(many.V_sub.isApprox(one.V_sub, 1e-15));
    REQUIRE(many.F_sub.isApprox(one.F_sub));
    REQUIRE(Eigen::MatrixXd(many.S - one.S).norm()
            < 1e-15 * Eigen::MatrixXd(one.S).norm());
}

TEST_CASE("loop_subdivide_n_times: n_passes = 2 matches the manual composition",
          "[shell][loop][subdivide][multipass]")
{
    const auto m  = make_octahedron();
    const auto s1 = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);
    const auto s2 = chladni::shell::loop::loop_subdivide_one_step(s1.V_sub, s1.F_sub);
    const auto two = chladni::shell::loop::loop_subdivide_n_times(m.V, m.F, 2);

    REQUIRE(two.V_sub.rows() == s2.V_sub.rows());
    REQUIRE(two.F_sub.rows() == s2.F_sub.rows());
    REQUIRE(two.V_sub.isApprox(s2.V_sub, 1e-12));
    REQUIRE(two.F_sub.isApprox(s2.F_sub));

    // Composed S: two-pass should map V -> V_sub_after_two_passes.
    Eigen::VectorXd v_flat(3 * m.V.rows());
    for (Eigen::Index i = 0; i < m.V.rows(); ++i) {
        for (int d = 0; d < 3; ++d) {
            v_flat(3 * i + d) = m.V(i, d);
        }
    }
    Eigen::VectorXd vsub_flat = two.S * v_flat;
    REQUIRE(vsub_flat.size() == 3 * two.V_sub.rows());
    for (Eigen::Index i = 0; i < two.V_sub.rows(); ++i) {
        for (int d = 0; d < 3; ++d) {
            REQUIRE_THAT(vsub_flat(3 * i + d),
                         WithinAbs(two.V_sub(i, d), 1e-12));
        }
    }
}

TEST_CASE("loop_subdivide_n_times: rigid translation is preserved through k passes",
          "[shell][loop][subdivide][multipass][constraint]")
{
    const auto m = make_octahedron();
    for (int k : {0, 1, 2, 3}) {
        CAPTURE(k);
        const auto sub = chladni::shell::loop::loop_subdivide_n_times(
            m.V, m.F, k);
        // Constant displacement (1, 2, 3) on every original vertex
        // must lift through S to the same constant displacement on
        // every subdivided vertex (Loop preserves constants in the
        // limit; the per-pass mass conservation composes).
        Eigen::VectorXd u(3 * m.V.rows());
        for (Eigen::Index i = 0; i < m.V.rows(); ++i) {
            u(3 * i + 0) = 1.0;
            u(3 * i + 1) = 2.0;
            u(3 * i + 2) = 3.0;
        }
        const Eigen::VectorXd u_sub = sub.S * u;
        REQUIRE(u_sub.size() == 3 * sub.V_sub.rows());
        for (Eigen::Index i = 0; i < sub.V_sub.rows(); ++i) {
            REQUIRE_THAT(u_sub(3 * i + 0), WithinAbs(1.0, 1e-12));
            REQUIRE_THAT(u_sub(3 * i + 1), WithinAbs(2.0, 1e-12));
            REQUIRE_THAT(u_sub(3 * i + 2), WithinAbs(3.0, 1e-12));
        }
    }
}

TEST_CASE("loop_subdivide_n_times: throws on negative n_passes",
          "[shell][loop][subdivide][multipass][validation]")
{
    const auto m = make_octahedron();
    REQUIRE_THROWS_AS(
        chladni::shell::loop::loop_subdivide_n_times(m.V, m.F, -1),
        std::invalid_argument);
}

TEST_CASE("loop_subdivide_one_step: rigid translation maps to itself",
          "[shell][loop][subdivide][constraint]")
{
    const auto m = make_octahedron();
    const auto sub = chladni::shell::loop::loop_subdivide_one_step(m.V, m.F);

    // Constant displacement (1, 2, 3) on every input vertex must map
    // through S to the same constant displacement on every subdivided
    // vertex (Loop subdivision preserves constants — partition-of-
    // unity property of the smoothing and edge rules combined).
    Eigen::VectorXd u(3 * m.V.rows());
    for (Eigen::Index i = 0; i < m.V.rows(); ++i) {
        u(3 * i + 0) = 1.0;
        u(3 * i + 1) = 2.0;
        u(3 * i + 2) = 3.0;
    }
    const Eigen::VectorXd u_sub = sub.S * u;
    REQUIRE(u_sub.size() == 3 * sub.V_sub.rows());
    for (Eigen::Index i = 0; i < sub.V_sub.rows(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(u_sub(3 * i + 0), WithinAbs(1.0, 1e-14));
        REQUIRE_THAT(u_sub(3 * i + 1), WithinAbs(2.0, 1e-14));
        REQUIRE_THAT(u_sub(3 * i + 2), WithinAbs(3.0, 1e-14));
    }
}
