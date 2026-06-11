/**
 * @file test_loop_boundary.cpp
 * @brief Tests for augment_for_loop_boundary (Cirak-Ortiz Eq. 54).
 *
 * The Schweitzer phantom-vertex augmentation closes the open fans at
 * boundary vertices so the regular Loop subdivision basis applies to
 * boundary patches. Tests use a small open cylinder (n_around = 8,
 * n_along = 2) where every rim vertex has valence 4.
 *
 * Properties verified:
 *  1. Vertex / face counts match the expected (n_real + n_phantom,
 *     n_real_faces + 2 * n_boundary_vertices).
 *  2. Phantom positions match the Eq. (54) formula
 *     @f$p = v_0 + v_1 - v_\text{int}@f$ for every boundary edge.
 *  3. Constraint matrix C has identity in its top n_real rows and
 *     exactly 3 nonzeros @f$(+1, +1, -1)@f$ in each phantom row.
 *  4. The augmented mesh's directed-edge-to-face map is 1-to-1 (no
 *     directed edge appears in two different faces — a manifold
 *     necessary condition).
 *  5. Every original boundary corner has valence 6 in the augmented
 *     mesh, and a closed CCW spoke walk around every original
 *     boundary corner returns 6 spokes.
 *  6. canonical_regular_dofs runs on a real triangle that touches the
 *     boundary, after augmentation. (Without augmentation the same
 *     call throws because the boundary corner has an open fan.)
 *  7. The function throws on a mesh with a valence-2 boundary corner
 *     (placeholder for the L.5c follow-up).
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace cs   = chladni::shell;
namespace csl  = chladni::shell::loop;
namespace cmsh = chladni::mesh;

namespace {

// Open cylinder: 2 boundary loops (top and bottom rim), all rim
// vertices have valence 4. n_around = 8, n_along = 2 keeps the test
// mesh small while still exercising the topology cleanly.
struct CylinderFixture {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    int             n_around;
    int             n_along;
};

CylinderFixture make_small_cylinder()
{
    CylinderFixture f;
    f.n_around = 8;
    f.n_along  = 2;
    auto m = cmsh::generate_cylinder(1.0, 1.0, f.n_around, f.n_along);
    f.V = std::move(m.V);
    f.F = std::move(m.F);
    return f;
}

}  // namespace

TEST_CASE("augment_for_loop_boundary: vertex / face counts on cylinder",
          "[shell][loop][boundary][counts]")
{
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    const Eigen::Index n_real_v       = fix.V.rows();
    const Eigen::Index n_real_f       = fix.F.rows();
    const Eigen::Index n_bvert_per_rim = fix.n_around;
    const Eigen::Index n_bvert         = 2 * n_bvert_per_rim;
    const Eigen::Index n_bedge         = n_bvert;  // closed loop

    REQUIRE(aug.n_real          == n_real_v);
    REQUIRE(aug.n_real_faces    == n_real_f);
    REQUIRE(aug.n_phantom       == n_bedge);
    REQUIRE(aug.n_phantom_faces == 2 * n_bvert);

    REQUIRE(aug.V_aug.rows() == n_real_v + n_bedge);
    REQUIRE(aug.V_aug.cols() == fix.V.cols());
    REQUIRE(aug.F_aug.rows() == n_real_f + 2 * n_bvert);
    REQUIRE(aug.F_aug.cols() == 3);

    REQUIRE(aug.C.rows() == 3 * (n_real_v + n_bedge));
    REQUIRE(aug.C.cols() == 3 * n_real_v);

    // V_aug.topRows(n_real) is exactly V; F_aug.topRows(n_real_faces) is F.
    REQUIRE((aug.V_aug.topRows(n_real_v) - fix.V).norm() == 0.0);
    REQUIRE((aug.F_aug.topRows(n_real_f) - fix.F).norm() == 0);
}

TEST_CASE("augment_for_loop_boundary: phantom positions match v0+v1-v_int",
          "[shell][loop][boundary][positions]")
{
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    const auto edges = cs::build_edges(fix.F);
    std::vector<Eigen::Index> bedge_idx;
    for (Eigen::Index e = 0; e < static_cast<Eigen::Index>(edges.size()); ++e)
    {
        if (edges[static_cast<std::size_t>(e)].is_boundary())
            bedge_idx.push_back(e);
    }
    REQUIRE(static_cast<Eigen::Index>(bedge_idx.size()) == aug.n_phantom);

    for (Eigen::Index k = 0; k < aug.n_phantom; ++k) {
        const auto& e = edges[static_cast<std::size_t>(
            bedge_idx[static_cast<std::size_t>(k)])];
        const Eigen::Index f_real =
            (e.face_left != -1) ? e.face_left : e.face_right;
        // Find the third vertex of the unique boundary face.
        Eigen::Index v_int = -1;
        for (int j = 0; j < 3; ++j) {
            const Eigen::Index w = fix.F(f_real, j);
            if (w != e.v0 && w != e.v1) { v_int = w; break; }
        }
        REQUIRE(v_int != -1);

        const Eigen::Vector3d expected =
            fix.V.row(e.v0).transpose()
          + fix.V.row(e.v1).transpose()
          - fix.V.row(v_int).transpose();
        const Eigen::Vector3d got =
            aug.V_aug.row(aug.n_real + k).transpose();
        for (int d = 0; d < 3; ++d) {
            REQUIRE(got(d) == Catch::Approx(expected(d)).margin(1e-13));
        }
    }
}

TEST_CASE("augment_for_loop_boundary: cylinder phantoms lie one ring outside",
          "[shell][loop][boundary][positions][cylinder]")
{
    // For a cylinder of radius R, length L, n_around N, n_along M, the
    // phantom across boundary edge {(i, 0), (i+1, 0)} has third vertex
    // (i+1, 1), so phantom = (R cos(theta_i), R sin(theta_i), -L/M).
    // Symmetrically for the top rim, phantom z = L + L/M.
    const double R = 1.0;
    const double L = 1.0;
    const int    N = 8;
    const int    M = 2;
    const auto m = cmsh::generate_cylinder(R, L, N, M);
    const auto aug = csl::augment_for_loop_boundary(m.V, m.F);

    const double dz = L / static_cast<double>(M);
    const double tol = 1e-13;

    // Each phantom z-coordinate must equal either -dz or L + dz.
    int n_below = 0;
    int n_above = 0;
    for (Eigen::Index k = 0; k < aug.n_phantom; ++k) {
        const double z = aug.V_aug(aug.n_real + k, 2);
        const double r = aug.V_aug.row(aug.n_real + k).head<2>().norm();
        REQUIRE(r == Catch::Approx(R).margin(tol));
        if (std::abs(z - (-dz)) < tol)            ++n_below;
        else if (std::abs(z - (L + dz)) < tol)    ++n_above;
        else FAIL("phantom z = " << z
                  << " is neither -dz nor L+dz");
    }
    REQUIRE(n_below == N);
    REQUIRE(n_above == N);
}

TEST_CASE("augment_for_loop_boundary: constraint matrix structure",
          "[shell][loop][boundary][constraints]")
{
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    // Top n_real * 3 rows: identity.
    const Eigen::SparseMatrix<double> top =
        aug.C.topRows(3 * aug.n_real);
    Eigen::SparseMatrix<double> id(3 * aug.n_real, 3 * aug.n_real);
    id.setIdentity();
    REQUIRE((Eigen::MatrixXd(top) - Eigen::MatrixXd(id)).norm()
            == Catch::Approx(0.0).margin(1e-15));

    // Bottom n_phantom * 3 rows: each row has exactly 3 nonzeros with
    // values (+1, +1, -1).
    const Eigen::Index dim_aug  = 3 * (aug.n_real + aug.n_phantom);
    const Eigen::SparseMatrix<double> bot =
        aug.C.bottomRows(dim_aug - 3 * aug.n_real);
    Eigen::Index n_phantom_rows = bot.rows();
    REQUIRE(n_phantom_rows == 3 * aug.n_phantom);

    Eigen::MatrixXd dense_bot = Eigen::MatrixXd(bot);
    for (Eigen::Index r = 0; r < dense_bot.rows(); ++r) {
        int n_pos = 0;
        int n_neg = 0;
        int n_other = 0;
        for (Eigen::Index c = 0; c < dense_bot.cols(); ++c) {
            const double x = dense_bot(r, c);
            if (x == 0.0) continue;
            if      (x ==  1.0) ++n_pos;
            else if (x == -1.0) ++n_neg;
            else                ++n_other;
        }
        REQUIRE(n_pos == 2);
        REQUIRE(n_neg == 1);
        REQUIRE(n_other == 0);
    }

    // Spot-check that applying C to a uniform translation gives the
    // same translation in every augmented DOF (rigid-translation
    // invariance is built in: phantom = v0 + v1 - v_int gives v0 + v1
    // - v_int = t + t - t = t when each real DOF is translated by t).
    Eigen::VectorXd u_real(3 * aug.n_real);
    for (Eigen::Index v = 0; v < aug.n_real; ++v) {
        u_real(3 * v + 0) = 1.5;
        u_real(3 * v + 1) = -2.5;
        u_real(3 * v + 2) = 0.7;
    }
    const Eigen::VectorXd u_aug = aug.C * u_real;
    for (Eigen::Index v = 0; v < aug.n_real + aug.n_phantom; ++v) {
        REQUIRE(u_aug(3 * v + 0) == Catch::Approx( 1.5).margin(1e-13));
        REQUIRE(u_aug(3 * v + 1) == Catch::Approx(-2.5).margin(1e-13));
        REQUIRE(u_aug(3 * v + 2) == Catch::Approx( 0.7).margin(1e-13));
    }
}

TEST_CASE("augment_for_loop_boundary: F_aug directed-edge map is 1-to-1",
          "[shell][loop][boundary][manifold]")
{
    // Each directed edge must appear in at most one face of F_aug,
    // otherwise the CCW topology walk inside canonical_regular_dofs
    // is ill-defined. (For a closed manifold mesh every directed edge
    // appears in exactly one face; for a manifold WITH boundary, the
    // boundary directed edges appear in zero faces. Either way, no
    // directed edge appears more than once.)
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index> de2f;
    for (Eigen::Index f = 0; f < aug.F_aug.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = aug.F_aug(f, k);
            const Eigen::Index b = aug.F_aug(f, (k + 1) % 3);
            const auto key = std::pair{a, b};
            const auto [it, inserted] = de2f.try_emplace(key, f);
            INFO("directed edge ("
                 << a << ", " << b << ") appears in faces "
                 << it->second << " and " << f);
            REQUIRE(inserted);
        }
    }
}

TEST_CASE("augment_for_loop_boundary: original boundary corners get closed valence-6 fans",
          "[shell][loop][boundary][fan]")
{
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    const auto edges_aug = cs::build_edges(aug.F_aug);
    const auto val_aug   = csl::vertex_valences(aug.V_aug.rows(), edges_aug);

    // Every original vertex (interior or boundary) should be
    // valence-6 in F_aug. Boundary vertices were valence-4 in F and
    // gained 2 phantom edges; interior vertices kept their valence.
    const auto edges_orig = cs::build_edges(fix.F);
    const auto val_orig   = csl::vertex_valences(fix.V.rows(), edges_orig);
    for (Eigen::Index v = 0; v < aug.n_real; ++v) {
        if (val_orig[static_cast<std::size_t>(v)] == 4) {
            REQUIRE(val_aug[static_cast<std::size_t>(v)] == 6);
        } else if (val_orig[static_cast<std::size_t>(v)] == 6) {
            REQUIRE(val_aug[static_cast<std::size_t>(v)] == 6);
        }
    }

    // Spot-check: the directed-edge-to-face map walk around an
    // original boundary corner closes after exactly 6 spokes. Pick the
    // first boundary vertex.
    Eigen::Index v_check = -1;
    for (Eigen::Index v = 0; v < aug.n_real; ++v) {
        if (val_orig[static_cast<std::size_t>(v)] == 4) {
            v_check = v;
            break;
        }
    }
    REQUIRE(v_check != -1);

    // Build directed-edge-to-face map and walk CCW.
    std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index> d2f;
    for (Eigen::Index f = 0; f < aug.F_aug.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = aug.F_aug(f, k);
            const Eigen::Index b = aug.F_aug(f, (k + 1) % 3);
            d2f[{a, b}] = f;
        }
    }

    // Find a starting neighbour of v_check.
    Eigen::Index start = -1;
    for (const auto& e : edges_aug) {
        if (e.v0 == v_check) { start = e.v1; break; }
        if (e.v1 == v_check) { start = e.v0; break; }
    }
    REQUIRE(start != -1);

    // Walk: at each step, find the unique face containing directed
    // edge (v_check, current); the next spoke is the third vertex.
    std::set<Eigen::Index> visited;
    Eigen::Index current = start;
    for (int step = 0; step < 8; ++step) {
        visited.insert(current);
        const auto it = d2f.find({v_check, current});
        REQUIRE(it != d2f.end());
        const Eigen::Index f = it->second;
        Eigen::Index next = -1;
        for (int k = 0; k < 3; ++k) {
            if (aug.F_aug(f, k) == v_check
                && aug.F_aug(f, (k + 1) % 3) == current)
            {
                next = aug.F_aug(f, (k + 2) % 3);
                break;
            }
        }
        REQUIRE(next != -1);
        if (next == start) {
            REQUIRE(visited.size() == 6);
            return;
        }
        current = next;
    }
    FAIL("walk around boundary vertex " << v_check
         << " did not close within 8 steps");
}

TEST_CASE("augment_for_loop_boundary: canonical_regular_dofs runs on a boundary patch after augmentation",
          "[shell][loop][boundary][canonical]")
{
    const auto fix = make_small_cylinder();
    const auto aug = csl::augment_for_loop_boundary(fix.V, fix.F);

    // Pick a real triangle whose first corner is a boundary vertex.
    const auto edges_orig = cs::build_edges(fix.F);
    const auto val_orig   = csl::vertex_valences(fix.V.rows(), edges_orig);
    Eigen::Index f_check = -1;
    for (Eigen::Index f = 0; f < fix.F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            if (val_orig[static_cast<std::size_t>(fix.F(f, k))] == 4) {
                f_check = f;
                break;
            }
        }
        if (f_check != -1) break;
    }
    REQUIRE(f_check != -1);

    // Build patch stencils on the augmented mesh; the first f_check
    // entry is the same triangle but with valences and ring resolved
    // against F_aug.
    const auto patches_aug =
        csl::build_patch_stencils(aug.V_aug, aug.F_aug);
    const auto& p = patches_aug[static_cast<std::size_t>(f_check)];

    // After augmentation each corner of a real boundary triangle has
    // valence 6 in F_aug.
    REQUIRE(p.corner_valences[0] == 6);
    REQUIRE(p.corner_valences[1] == 6);
    REQUIRE(p.corner_valences[2] == 6);

    // canonical_regular_dofs walks the augmented mesh and should
    // succeed (the fan is closed). Returns 12 distinct indices.
    const auto dofs = csl::canonical_regular_dofs(p, aug.F_aug);
    std::set<Eigen::Index> uniq(dofs.begin(), dofs.end());
    REQUIRE(uniq.size() == 12);
    // At least one phantom (index >= n_real) appears in the slot
    // ordering, since the original triangle has a boundary corner.
    bool has_phantom = false;
    for (Eigen::Index d : dofs) {
        if (d >= aug.n_real) { has_phantom = true; break; }
    }
    REQUIRE(has_phantom);
}

// ---------------------------------------------------------------------------
// L.5c.4: valence-2 boundary corner support.
//
// Test fixture is a single unit-square quad with one diagonal — the
// smallest mesh exhibiting valence-2 corners. With the diagonal V0-V2,
// vertex V1 (1, 0) and V3 (0, 1) have only 2 incident edges each
// (their two boundary edges, sharing only the unique single triangle
// at that corner). V0 and V2 are valence-3 (the diagonal endpoints).
// ---------------------------------------------------------------------------

namespace {

struct SingleQuad {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

SingleQuad make_single_quad()
{
    SingleQuad q;
    q.V.resize(4, 3);
    q.V << 0.0, 0.0, 0.0,
           1.0, 0.0, 0.0,
           1.0, 1.0, 0.0,
           0.0, 1.0, 0.0;
    q.F.resize(2, 3);
    q.F << 0, 1, 2,
           0, 2, 3;
    return q;
}

}  // namespace

TEST_CASE("augment_for_loop_boundary: single-quad accepts valence-2 corners",
          "[shell][loop][boundary][corner][valence2]")
{
    const auto q = make_single_quad();
    REQUIRE_NOTHROW(csl::augment_for_loop_boundary(q.V, q.F));
    const auto aug = csl::augment_for_loop_boundary(q.V, q.F);

    // Boundary edges: (V0,V1), (V1,V2), (V2,V3), (V3,V0). 4 edge phantoms.
    // Corner phantoms: V0 valence-3 -> 1; V1 valence-2 -> 2; V2 valence-3 -> 1;
    // V3 valence-2 -> 2. Total: 6 corner phantoms.
    REQUIRE(aug.n_real    == 4);
    REQUIRE(aug.n_phantom == 4 + 6);
    REQUIRE(aug.V_aug.rows() == 4 + 10);

    // Phantom faces: 4 across-edge + (1+2+1+2) corner-fills = 4 + 6 = 10.
    // Wait, valence-3 corner = 2 fills, valence-2 corner = 3 fills.
    // 2 valence-3 corners * 2 fills + 2 valence-2 corners * 3 fills
    // + 4 across-edge phantom triangles = 4 + 6 + 4 = 14.
    REQUIRE(aug.n_phantom_faces == 14);
}

TEST_CASE("augment_for_loop_boundary: single-quad valence-2 phantom positions match reflection rule",
          "[shell][loop][boundary][corner][valence2]")
{
    const auto q = make_single_quad();
    const auto aug = csl::augment_for_loop_boundary(q.V, q.F);

    // Corner phantom rows are stored after the 4 across-edge phantoms,
    // i.e. starting at row n_real + 4 = 8. They appear in ascending
    // real-vertex order, with valence-2 vertices contributing 2
    // consecutive rows and valence-3 vertices contributing 1.
    //
    // Boundary loop CCW around the quad: V0 -> V1 -> V2 -> V3 -> V0.
    // Per-vertex (valence, in-neighbour, out-neighbour, interior u for v-3):
    //   V0 (val 3, in=V3, out=V1, u=V2) -> 1 corner phantom
    //   V1 (val 2, in=V0, out=V2)       -> 2 corner phantoms (p_A, p_B)
    //   V2 (val 3, in=V1, out=V3, u=V0) -> 1 corner phantom
    //   V3 (val 2, in=V2, out=V0)       -> 2 corner phantoms (p_A, p_B)
    //
    // Reflection rule for valence-2: p_A = 2v - b_out, p_B = 2v - b_in.
    const Eigen::Index r = aug.n_real + 4;  // start of corner phantoms

    // V0 corner phantom (valence-3): v + b_in + b_out - 2 u
    //   = V0 + V3 + V1 - 2 V2 = (0,0,0)+(0,1,0)+(1,0,0)-2*(1,1,0) = (-1,-1,0).
    REQUIRE(aug.V_aug.row(r + 0).isApprox(
        Eigen::RowVector3d(-1.0, -1.0, 0.0), 1e-12));

    // V1 valence-2: p_A = 2*V1 - V2 = (1,-1,0); p_B = 2*V1 - V0 = (2,0,0).
    REQUIRE(aug.V_aug.row(r + 1).isApprox(
        Eigen::RowVector3d( 1.0, -1.0, 0.0), 1e-12));
    REQUIRE(aug.V_aug.row(r + 2).isApprox(
        Eigen::RowVector3d( 2.0,  0.0, 0.0), 1e-12));

    // V2 corner phantom (valence-3): V2 + V1 + V3 - 2 V0 = (1,1)+(1,0)+(0,1)-(0,0)
    // = (2, 2, 0).
    REQUIRE(aug.V_aug.row(r + 3).isApprox(
        Eigen::RowVector3d( 2.0,  2.0, 0.0), 1e-12));

    // V3 valence-2: p_A = 2*V3 - V0 = (0,2,0); p_B = 2*V3 - V2 = (-1,1,0).
    REQUIRE(aug.V_aug.row(r + 4).isApprox(
        Eigen::RowVector3d( 0.0,  2.0, 0.0), 1e-12));
    REQUIRE(aug.V_aug.row(r + 5).isApprox(
        Eigen::RowVector3d(-1.0,  1.0, 0.0), 1e-12));
}

TEST_CASE("augment_for_loop_boundary: single-quad augmented valences are all 6",
          "[shell][loop][boundary][corner][valence2][fan]")
{
    const auto q = make_single_quad();
    const auto aug = csl::augment_for_loop_boundary(q.V, q.F);

    const auto edges_aug = cs::build_edges(aug.F_aug);
    const auto vals_aug  = csl::vertex_valences(aug.V_aug.rows(), edges_aug);

    // All 4 original vertices must close to valence 6 in the augmented
    // mesh: 2 boundary real + (valence-2: 0 interior real, 2 phantom
    // edge, 2 phantom corner) = 6, or (valence-3: 1 interior real,
    // 2 phantom edge, 1 phantom corner) = 6.
    for (int v = 0; v < 4; ++v) {
        CAPTURE(v);
        REQUIRE(vals_aug[v] == 6);
    }
}

TEST_CASE("augment_for_loop_boundary: single-quad augmented mesh is manifold",
          "[shell][loop][boundary][corner][valence2][manifold]")
{
    const auto q = make_single_quad();
    const auto aug = csl::augment_for_loop_boundary(q.V, q.F);

    std::map<std::pair<Eigen::Index, Eigen::Index>, int> dedge_count;
    for (Eigen::Index f = 0; f < aug.F_aug.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = aug.F_aug(f, k);
            const Eigen::Index b = aug.F_aug(f, (k + 1) % 3);
            ++dedge_count[{a, b}];
        }
    }
    for (const auto& [edge, count] : dedge_count) {
        CAPTURE(edge.first, edge.second);
        REQUIRE(count == 1);
    }
}

TEST_CASE("augment_for_loop_boundary: single-quad constraint matrix preserves rigid translation",
          "[shell][loop][boundary][corner][valence2][constraint]")
{
    const auto q = make_single_quad();
    const auto aug = csl::augment_for_loop_boundary(q.V, q.F);

    Eigen::VectorXd u(3 * aug.n_real);
    for (Eigen::Index i = 0; i < aug.n_real; ++i) {
        u(3 * i + 0) = 1.0;
        u(3 * i + 1) = 2.0;
        u(3 * i + 2) = 3.0;
    }
    const Eigen::VectorXd Cu = aug.C * u;
    REQUIRE(Cu.size() == 3 * (aug.n_real + aug.n_phantom));
    for (Eigen::Index i = 0; i < aug.n_real + aug.n_phantom; ++i) {
        CAPTURE(i);
        REQUIRE(Cu(3 * i + 0) == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(Cu(3 * i + 1) == Catch::Approx(2.0).margin(1e-12));
        REQUIRE(Cu(3 * i + 2) == Catch::Approx(3.0).margin(1e-12));
    }
}

TEST_CASE("assemble_stiffness_loop: single-quad ships through end-to-end",
          "[shell][loop][boundary][corner][valence2][global_k]")
{
    const auto q = make_single_quad();
    cs::ShellMaterial mat;
    mat.k_L           = 1.0e6;
    mat.k_B           = 1.0e3;
    mat.poisson_ratio = 0.3;

    REQUIRE_NOTHROW(csl::assemble_stiffness_loop(q.V, q.F, mat));
    const auto K = csl::assemble_stiffness_loop(q.V, q.F, mat);

    REQUIRE(K.rows() == 3 * q.V.rows());
    REQUIRE(K.cols() == K.rows());

    Eigen::SparseMatrix<double> Kt = K.transpose();
    const double scale = Eigen::MatrixXd(K).norm();
    REQUIRE(scale > 0.0);
    REQUIRE(Eigen::MatrixXd(K - Kt).norm() <= 1e-9 * scale);

    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd t = Eigen::VectorXd::Zero(3 * q.V.rows());
        for (Eigen::Index v = 0; v < q.V.rows(); ++v) {
            t(3 * v + axis) = 1.0;
        }
        REQUIRE((K * t).norm() <= 1e-9 * scale);
    }
}

// ---------------------------------------------------------------------------
// L.5c.1: valence-3 boundary corner support.
//
// Test fixture is a "hex fan": one valence-6 interior vertex (V3) at
// the centre of a flat hexagonal flap, with 6 valence-3 boundary
// corners around it. This is exactly the topology produced by
// removing the 2 valence-2 corners from a 2x2 grid (chamfering) and
// is a stand-in for the corners of a flat plate generator.
//
// Vertex IDs and (x, y, z):
//   V0 = (0, 0, 0)       boundary corner, valence 3
//   V1 = (1, 0, 0)       boundary corner, valence 3
//   V2 = (0, 1, 0)       boundary corner, valence 3
//   V3 = (1, 1, 0)       interior, valence 6
//   V4 = (2, 1, 0)       boundary corner, valence 3
//   V5 = (1, 2, 0)       boundary corner, valence 3
//   V6 = (2, 2, 0)       boundary corner, valence 3
// Triangles (CCW):
//   (V0, V1, V3), (V0, V3, V2),
//   (V1, V4, V3), (V2, V3, V5),
//   (V3, V4, V6), (V3, V6, V5).
// Boundary loop (CCW around the hexagonal flap, viewed from +z):
//   V0 -> V1 -> V4 -> V6 -> V5 -> V2 -> V0.
// ---------------------------------------------------------------------------

namespace {

struct HexFan {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

HexFan make_hex_fan()
{
    HexFan h;
    h.V.resize(7, 3);
    h.V << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           1, 1, 0,
           2, 1, 0,
           1, 2, 0,
           2, 2, 0;
    h.F.resize(6, 3);
    h.F << 0, 1, 3,
           0, 3, 2,
           1, 4, 3,
           2, 3, 5,
           3, 4, 6,
           3, 6, 5;
    return h;
}

}  // namespace

TEST_CASE("augment_for_loop_boundary: hex-fan accepts 6 valence-3 corners",
          "[shell][loop][boundary][corner][valence3]")
{
    const auto h = make_hex_fan();
    REQUIRE_NOTHROW(csl::augment_for_loop_boundary(h.V, h.F));
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);

    // 6 boundary edges + 6 valence-3 corners => 12 phantom vertices.
    REQUIRE(aug.n_real    == 7);
    REQUIRE(aug.n_phantom == 12);
    REQUIRE(aug.V_aug.rows() == 7 + 12);

    // Phantom faces: 1 across-edge per boundary edge (6) + 2 corner-fill
    // per valence-3 corner (12) = 18.
    REQUIRE(aug.n_phantom_faces == 18);
    REQUIRE(aug.F_aug.rows() == 6 + 18);
}

TEST_CASE("augment_for_loop_boundary: hex-fan corner phantom positions match parallelogram rule",
          "[shell][loop][boundary][corner][valence3]")
{
    const auto h = make_hex_fan();
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);

    // Corner phantoms are stored after the 6 across-edge phantoms,
    // i.e. rows [n_real + 6, n_real + 12) of V_aug. They appear in
    // ascending real-vertex order. For each valence-3 corner v with
    // boundary neighbours b_in, b_out and unique interior neighbour
    // u = V3 (= row 3), the position is v + b_in + b_out - 2 u.
    // The boundary loop CCW is V0 -> V1 -> V4 -> V6 -> V5 -> V2 -> V0.
    const Eigen::Vector3d V3 = h.V.row(3);
    auto expected_corner =
        [&](int v, int b_in, int b_out) {
            return (h.V.row(v) + h.V.row(b_in) + h.V.row(b_out)
                    - 2.0 * V3.transpose()).eval();
        };

    // Corner-row offset.
    const Eigen::Index r0 = aug.n_real + 6;

    // V0: in-neighbour V2, out-neighbour V1 (CCW loop ... V2 -> V0 -> V1).
    REQUIRE(aug.V_aug.row(r0 + 0).isApprox(expected_corner(0, 2, 1), 1e-12));
    // V1: in V0, out V4.
    REQUIRE(aug.V_aug.row(r0 + 1).isApprox(expected_corner(1, 0, 4), 1e-12));
    // V2: in V5, out V0.
    REQUIRE(aug.V_aug.row(r0 + 2).isApprox(expected_corner(2, 5, 0), 1e-12));
    // V4: in V1, out V6.
    REQUIRE(aug.V_aug.row(r0 + 3).isApprox(expected_corner(4, 1, 6), 1e-12));
    // V5: in V6, out V2.
    REQUIRE(aug.V_aug.row(r0 + 4).isApprox(expected_corner(5, 6, 2), 1e-12));
    // V6: in V4, out V5.
    REQUIRE(aug.V_aug.row(r0 + 5).isApprox(expected_corner(6, 4, 5), 1e-12));
}

TEST_CASE("augment_for_loop_boundary: hex-fan augmented valences are all 6",
          "[shell][loop][boundary][corner][valence3][fan]")
{
    const auto h = make_hex_fan();
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);

    const auto edges_aug = cs::build_edges(aug.F_aug);
    const auto vals_aug  = csl::vertex_valences(aug.V_aug.rows(), edges_aug);

    // Original interior vertex V3 was valence 6 already.
    REQUIRE(vals_aug[3] == 6);
    // Each of the 6 valence-3 corners must close to valence 6 in the
    // augmented mesh: 2 real boundary + 1 real interior + 2 phantom
    // edge + 1 phantom corner = 6.
    for (int v : {0, 1, 2, 4, 5, 6}) {
        CAPTURE(v);
        REQUIRE(vals_aug[v] == 6);
    }
}

TEST_CASE("augment_for_loop_boundary: hex-fan augmented mesh is manifold",
          "[shell][loop][boundary][corner][valence3][manifold]")
{
    const auto h = make_hex_fan();
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);

    // Every directed edge appears in exactly one face — necessary for
    // canonical_regular_dofs's CCW topology walk to terminate.
    std::map<std::pair<Eigen::Index, Eigen::Index>, int> dedge_count;
    for (Eigen::Index f = 0; f < aug.F_aug.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = aug.F_aug(f, k);
            const Eigen::Index b = aug.F_aug(f, (k + 1) % 3);
            ++dedge_count[{a, b}];
        }
    }
    for (const auto& [edge, count] : dedge_count) {
        CAPTURE(edge.first, edge.second);
        REQUIRE(count == 1);
    }
}

TEST_CASE("augment_for_loop_boundary: canonical_regular_dofs runs on hex-fan real triangles",
          "[shell][loop][boundary][corner][valence3][canonical]")
{
    const auto h = make_hex_fan();
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);
    const auto patches =
        csl::build_patch_stencils(aug.V_aug, aug.F_aug);

    // Each real triangle (rows 0..5 of F_aug) should now have all 3
    // augmented corner valences equal to 6 and canonical_regular_dofs
    // should return 12 valid indices.
    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        CAPTURE(f);
        const auto& p = patches[static_cast<std::size_t>(f)];
        REQUIRE(p.corner_valences[0] == 6);
        REQUIRE(p.corner_valences[1] == 6);
        REQUIRE(p.corner_valences[2] == 6);
        REQUIRE_NOTHROW(csl::canonical_regular_dofs(p, aug.F_aug));
    }
}

TEST_CASE("augment_for_loop_boundary: hex-fan constraint matrix preserves rigid translation",
          "[shell][loop][boundary][corner][valence3][constraint]")
{
    const auto h = make_hex_fan();
    const auto aug = csl::augment_for_loop_boundary(h.V, h.F);

    // A constant displacement (1, 2, 3) on every real vertex must lift
    // through C to the same constant displacement on every augmented
    // vertex (sum of constraint coefficients is 1 for both phantom
    // types).
    Eigen::VectorXd u(3 * aug.n_real);
    for (Eigen::Index i = 0; i < aug.n_real; ++i) {
        u(3 * i + 0) = 1.0;
        u(3 * i + 1) = 2.0;
        u(3 * i + 2) = 3.0;
    }
    const Eigen::VectorXd Cu = aug.C * u;
    REQUIRE(Cu.size() == 3 * (aug.n_real + aug.n_phantom));
    for (Eigen::Index i = 0; i < aug.n_real + aug.n_phantom; ++i) {
        CAPTURE(i);
        REQUIRE(Cu(3 * i + 0) == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(Cu(3 * i + 1) == Catch::Approx(2.0).margin(1e-12));
        REQUIRE(Cu(3 * i + 2) == Catch::Approx(3.0).margin(1e-12));
    }
}

TEST_CASE("assemble_stiffness_loop: hex-fan ships through end-to-end",
          "[shell][loop][boundary][corner][valence3][global_k]")
{
    const auto h = make_hex_fan();
    cs::ShellMaterial mat;
    mat.k_L           = 1.0e6;
    mat.k_B           = 1.0e3;
    mat.poisson_ratio = 0.3;

    REQUIRE_NOTHROW(csl::assemble_stiffness_loop(h.V, h.F, mat));
    const auto K = csl::assemble_stiffness_loop(h.V, h.F, mat);

    REQUIRE(K.rows() == 3 * h.V.rows());
    REQUIRE(K.cols() == K.rows());

    // Symmetry.
    Eigen::SparseMatrix<double> Kt = K.transpose();
    const double scale = Eigen::MatrixXd(K).norm();
    REQUIRE(scale > 0.0);
    REQUIRE(Eigen::MatrixXd(K - Kt).norm() <= 1e-9 * scale);

    // Rigid translation along each axis annihilates K.
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd t = Eigen::VectorXd::Zero(3 * h.V.rows());
        for (Eigen::Index v = 0; v < h.V.rows(); ++v) {
            t(3 * v + axis) = 1.0;
        }
        REQUIRE((K * t).norm() <= 1e-9 * scale);
    }
}
