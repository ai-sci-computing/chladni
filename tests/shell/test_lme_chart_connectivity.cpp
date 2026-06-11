/**
 * @file test_lme_chart_connectivity.cpp
 * @brief Chart connectivity prune — enforce the topological-disk
 *        property of LME charts by construction.
 *
 * The paper's value-based neighbourhoods (Millán 2011 Eq. 2) are pure
 * Euclidean metric balls; nothing prevents such a ball on a sampled
 * surface from containing several geodesically disconnected
 * components (e.g. two nearby sheets of a fold). The paper handles
 * this BY ASSUMPTION, not by construction: applicability "depends
 * crucially ... on the density of the sampling relative to the
 * feature size" (their §5) — under which the surface cannot fold back
 * within a few node spacings and the ball IS a disk. Our charts must
 * serve arbitrary user meshes, so the geometric chart selection
 * (Q-D6) restricts the ball to the anchor's geodesically CONNECTED
 * component: a BFS through the mesh adjacency restricted to the
 * selected node set, dropping unreachable islands.
 *
 * Three layers of coverage:
 *
 * 1. @c lme::prune_to_anchor_component unit spec on a hand-built
 *    path graph (islands dropped, order preserved, anchor kept).
 *
 * 2. @c lme::extract_chart_neighbourhood unit spec (the production
 *    chart constructor: in-ball BFS = connected component of the
 *    Eq. 2 ball, NOT hop-bounded, with the 2-ring fallback and the
 *    nearest-cap + re-prune).
 *
 * 3. Integration on a FOLDED-RIBBON mesh: two parallel 10x3 strips
 *    (z=0 and z=0.5, in-plane spacing h=1) joined only at the x=9 end
 *    through a far-out bridge at x=11.5. For a mid-sheet anchor the
 *    opposite sheet is Euclidean-near (0.5 << the ~4.7h chart ball)
 *    and within the BFS-8 hop guard (via the bridge), while the
 *    bridge nodes themselves sit OUTSIDE the anchor's ball — so the
 *    ball ∩ k-ring chart contains the opposite sheet as a
 *    disconnected island. Without the prune, far cross-sheet K/M
 *    couplings appear (charts anchored at x≈6 span A(x≥1.3) and
 *    B(x≥7)); with it, every chart anchored away from the bend is
 *    sheet-pure and the far cross-sheet blocks are STRUCTURALLY zero,
 *    while the legitimate bend coupling (through charts whose ball
 *    contains the bridge, keeping the set connected) must survive.
 */

#include <chladni/analytical/plate.hpp>
#include <chladni/analytical/shell.hpp>
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <vector>

using chladni::shell::LMEAssembler;

namespace {

chladni::IsotropicMaterial steel_033()
{
    return {.youngs_modulus = 200.0e9,
            .poisson_ratio  = 0.33,
            .density        = 7850.0};
}

/// Folded-ribbon fixture: sheet A (z=0) and sheet B (z=0.5), each a
/// 14x3 unit grid (x=0..13, y=0..2), joined at the x=13 end through
/// two bridge rows at x=15.5 (z=0 and z=0.5). Quad strips are split
/// so NO triangle contains both an A(13,·) and a B(13,·) vertex — the
/// sheets connect only through the bridge nodes. The 2.5-long bridge
/// edges inflate h_a (and hence the chart ball) of the x=13 rim
/// nodes, so the sheets must be long enough that those wide-but-
/// connected bend charts cannot reach the far assertion zones.
/// Indexing:
///   A(x,y) = 3x + y            (0..41)
///   B(x,y) = 42 + 3x + y       (42..83)
///   b1(y)  = 84 + y  (z=0)     (84..86)
///   b2(y)  = 87 + y  (z=0.5)   (87..89)
struct FoldedRibbon {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    static constexpr int nx = 14;
    static constexpr int ny = 3;

    static int iA(int x, int y) { return 3 * x + y; }
    static int iB(int x, int y) { return 42 + 3 * x + y; }
    static int ib1(int y) { return 84 + y; }
    static int ib2(int y) { return 87 + y; }

    FoldedRibbon()
    {
        V.resize(90, 3);
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y) {
                V.row(iA(x, y)) << x, y, 0.0;
                V.row(iB(x, y)) << x, y, 0.5;
            }
        for (int y = 0; y < ny; ++y) {
            V.row(ib1(y)) << 15.5, y, 0.0;
            V.row(ib2(y)) << 15.5, y, 0.5;
        }

        std::vector<Eigen::RowVector3i> tris;
        auto quad = [&](int v00, int v10, int v11, int v01) {
            tris.emplace_back(Eigen::RowVector3i{v00, v10, v11});
            tris.emplace_back(Eigen::RowVector3i{v00, v11, v01});
        };
        // Sheet grids. The whole ribbon is one parametric strip
        // A0..A9 → b1 → b2 → B9..B0, so sheet B winds with x
        // DECREASING to keep a globally consistent orientation
        // (each interior edge used once per direction).
        for (int x = 0; x + 1 < nx; ++x)
            for (int y = 0; y + 1 < ny; ++y) {
                quad(iA(x, y), iA(x + 1, y), iA(x + 1, y + 1), iA(x, y + 1));
                quad(iB(x + 1, y), iB(x, y), iB(x, y + 1), iB(x + 1, y + 1));
            }
        // Bend: A(13,·) → b1 → b2 → B(13,·) quad strips.
        for (int y = 0; y + 1 < ny; ++y) {
            quad(iA(13, y), ib1(y), ib1(y + 1), iA(13, y + 1));
            quad(ib1(y), ib2(y), ib2(y + 1), ib1(y + 1));
            quad(ib2(y), iB(13, y), iB(13, y + 1), ib2(y + 1));
        }
        F.resize(static_cast<Eigen::Index>(tris.size()), 3);
        for (std::size_t t = 0; t < tris.size(); ++t)
            F.row(static_cast<Eigen::Index>(t)) = tris[t];
    }
};

/// Max |entry| over the 3x3 (i, j) DOF block of a sparse matrix.
double block_max_abs(const Eigen::SparseMatrix<double>& K, int i, int j)
{
    double m = 0.0;
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
            m = std::max(m, std::abs(K.coeff(3 * i + c, 3 * j + d)));
    return m;
}

}  // namespace

TEST_CASE("lme::prune_to_anchor_component — islands dropped, order kept",
          "[lme][chart][connectivity]")
{
    // Path graph 0-1-2-3-4-5-6. Selected set {0,1,2,5,6}: the
    // connecting path 3-4 is NOT selected, so {5,6} is an island
    // relative to anchor 0.
    const std::vector<std::vector<int>> adj{
        {1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5}};

    const std::vector<int> kept = chladni::shell::lme::
        prune_to_anchor_component(0, {0, 1, 2, 5, 6}, adj);
    REQUIRE(kept == std::vector<int>{0, 1, 2});

    // Anchored at the island instead: {5,6} survive, {0,1,2} drop.
    const std::vector<int> kept5 = chladni::shell::lme::
        prune_to_anchor_component(5, {0, 1, 2, 5, 6}, adj);
    REQUIRE(kept5 == std::vector<int>{5, 6});

    // Fully connected selection is a no-op (input order preserved).
    const std::vector<int> all = chladni::shell::lme::
        prune_to_anchor_component(3, {1, 2, 3, 4, 5}, adj);
    REQUIRE(all == std::vector<int>{1, 2, 3, 4, 5});

    // Degenerate: anchor alone.
    const std::vector<int> solo = chladni::shell::lme::
        prune_to_anchor_component(0, {0, 5}, adj);
    REQUIRE(solo == std::vector<int>{0});
}

TEST_CASE("lme::extract_chart_neighbourhood — in-ball BFS spec",
          "[lme][chart][connectivity]")
{
    // Path graph 0-1-2-...-12 with unit spacing along x.
    Eigen::MatrixXd P(13, 3);
    std::vector<std::vector<int>> adj(13);
    for (int i = 0; i < 13; ++i) {
        P.row(i) << i, 0.0, 0.0;
        if (i > 0) {
            adj[static_cast<std::size_t>(i)].push_back(i - 1);
            adj[static_cast<std::size_t>(i - 1)].push_back(i);
        }
    }
    using chladni::shell::lme::extract_chart_neighbourhood;

    // Plain ball: radius 3.5 from node 0 keeps 0..3.
    REQUIRE(extract_chart_neighbourhood(0, 3.5, P, adj, 1, 100)
            == std::vector<int>{0, 1, 2, 3});

    // NOT hop-bounded (unlike the retired guard_depth=8 BFS): radius
    // 10.5 reaches node 10, which is 10 hops out.
    REQUIRE(extract_chart_neighbourhood(0, 10.5, P, adj, 1, 100).size()
            == 11);

    // Island exclusion: node 12 moved Euclidean-near the anchor but
    // still only reachable through the full chain (which exits the
    // ball) — the in-ball BFS cannot reach it.
    Eigen::MatrixXd Pfold = P;
    Pfold.row(12) << 0.5, 0.4, 0.0;  // |Pfold12 - P0| = 0.64 < 1.5
    const auto fold = extract_chart_neighbourhood(0, 1.5, Pfold, adj, 1, 100);
    REQUIRE(fold == std::vector<int>{0, 1});

    // Minimum-chart fallback: a sub-spacing radius keeps only the
    // anchor, so the plain 2-ring is returned instead.
    const auto tiny = extract_chart_neighbourhood(6, 0.5, P, adj, 7, 100);
    REQUIRE(tiny.size() == 5);  // {6, 5, 7, 4, 8}
    for (int v : {4, 5, 6, 7, 8})
        REQUIRE(std::find(tiny.begin(), tiny.end(), v) != tiny.end());

    // Nearest-cap: wide ball capped to 5 keeps the 5 nearest, still
    // connected and anchored.
    const auto capped = extract_chart_neighbourhood(6, 12.5, P, adj, 1, 5);
    REQUIRE(capped.size() == 5);
    REQUIRE(std::find(capped.begin(), capped.end(), 6) != capped.end());
    for (int v : capped) REQUIRE(std::abs(v - 6) <= 2);
}

TEST_CASE("lme::extract_chart_neighbourhood_directional — anisotropic spec",
          "[lme][chart][connectivity]")
{
    using chladni::shell::lme::extract_chart_neighbourhood_directional;

    // Anisotropic grid strip: 9 columns spaced 2.4 apart in x, 3 rows
    // spaced 1 apart in y (an aspect-2.4 cylinder-like patch,
    // flattened). Anchor = centre node (4,1).
    constexpr int nxg = 9, nyg = 3;
    constexpr double ax = 2.4;
    auto id = [&](int x, int y) { return x * nyg + y; };
    Eigen::MatrixXd P(nxg * nyg, 3);
    std::vector<std::vector<int>> adj(
        static_cast<std::size_t>(nxg * nyg));
    auto link = [&](int u, int v) {
        adj[static_cast<std::size_t>(u)].push_back(v);
        adj[static_cast<std::size_t>(v)].push_back(u);
    };
    for (int x = 0; x < nxg; ++x)
        for (int y = 0; y < nyg; ++y) {
            P.row(id(x, y)) << ax * x, y, 0.0;
            if (x > 0) link(id(x - 1, y), id(x, y));
            if (y > 0) link(id(x, y - 1), id(x, y));
        }
    const int anchor = id(4, 1);
    std::vector<Eigen::Vector3d> edges{
        {ax, 0, 0}, {-ax, 0, 0}, {0, 1, 0}, {0, -1, 0}};

    // mult=2.2: reach ~2.2 spacings in EACH direction — 2 columns in
    // x (distance up to 2·2.4=4.8 ≤ 2.2·2.4=5.28) but only the 1-wide
    // y rows exist. An isotropic ball of ANY radius cannot produce
    // this set: covering 2 columns (4.8) would swallow every y row
    // 4.8 wide if rows existed there.
    const auto sel =
        extract_chart_neighbourhood_directional(
            anchor, 2.2, edges, P, adj, 1, 100);
    for (int v : sel) {
        const int x = v / nyg, y = v % nyg;
        INFO("kept node x=" << x << " y=" << y);
        REQUIRE(std::abs(x - 4) <= 2);  // ≤2.2 axial spacings
        REQUIRE(std::abs(y - 1) <= 2);  // all rows in range (≤2 ≤ 2.2)
    }
    // Both extremes present: 2 columns out axially AND the corner
    // (2 columns + 1 row, mixed direction).
    REQUIRE(std::find(sel.begin(), sel.end(), id(6, 1)) != sel.end());
    REQUIRE(std::find(sel.begin(), sel.end(), id(2, 1)) != sel.end());
    REQUIRE(std::find(sel.begin(), sel.end(), id(4, 0)) != sel.end());

    // Flat-sheet fold exclusion with NO epsilon: a node straight off
    // the plane is metrically out of range (h_a(normal) = 0), even
    // when Euclidean-near and graph-adjacent.
    Eigen::MatrixXd Pz = P;
    const int off = id(0, 0);
    Pz.row(off) << ax * 4, 1.0, 0.3;  // 0.3 straight above the anchor
    link(off, anchor);                // even directly graph-adjacent
    const auto selz = extract_chart_neighbourhood_directional(
        anchor, 2.2, edges, Pz, adj, 1, 100);
    REQUIRE(std::find(selz.begin(), selz.end(), off) == selz.end());
}

TEST_CASE("lme::extract_chart_neighbourhood_intrinsic — geodesic spec",
          "[lme][chart][connectivity]")
{
    using chladni::shell::lme::extract_chart_neighbourhood_intrinsic;

    // Hairpin path: nodes 0..6 march along +x at unit spacing, then
    // the chain folds back 0.25 above itself, 7..12 marching -x. Node
    // 12 sits 0.25 straight above node 1: Euclidean-near, but its
    // INTRINSIC distance is ~11.3 along the chain.
    Eigen::MatrixXd P(13, 3);
    std::vector<std::vector<int>> adj(13);
    for (int i = 0; i < 13; ++i) {
        if (i <= 6) P.row(i) << i, 0.0, 0.0;
        else        P.row(i) << 12 - i + 1, 0.0, 0.25;
        if (i > 0) {
            adj[static_cast<std::size_t>(i)].push_back(i - 1);
            adj[static_cast<std::size_t>(i - 1)].push_back(i);
        }
    }

    // Geodesic radius 3.2 from node 0: nodes 0..3 — the fold sheet is
    // intrinsically out of range with NO connectivity machinery, even
    // though node 12 is only 0.27 away through ambient space.
    const auto sel =
        extract_chart_neighbourhood_intrinsic(0, 3.2, P, adj, 1, 100);
    REQUIRE(sel == std::vector<int>{0, 1, 2, 3});

    // Radius spanning the bend: from node 0, radius 8.5 reaches along
    // the chain through the fold onto the upper sheet (legitimate —
    // geodesically connected): 0..6 plus nodes 7/8/9 at intrinsic
    // 6.25/7.25/8.25.
    const auto far8 =
        extract_chart_neighbourhood_intrinsic(0, 8.5, P, adj, 1, 100);
    REQUIRE(far8.size() == 10);
    REQUIRE(far8.front() == 0);

    // Nearest-cap keeps the intrinsically nearest, not the
    // Euclidean-nearest: capped to 4 from node 0 keeps 0..3 — node 12
    // (Euclidean distance 1.03, nearer than node 2) must NOT displace
    // them.
    const auto capped =
        extract_chart_neighbourhood_intrinsic(0, 100.0, P, adj, 1, 4);
    REQUIRE(capped == std::vector<int>{0, 1, 2, 3});

    // Minimum-chart fallback: sub-spacing radius -> plain 2-ring.
    const auto tiny =
        extract_chart_neighbourhood_intrinsic(6, 0.4, P, adj, 5, 100);
    REQUIRE(tiny.size() == 5);  // {6, 5, 7, 4, 8}
}

TEST_CASE("SME chart-unification A/B — k-ring vs connected metric ball",
          "[.diag][lme][sme][chart][sme_chart_ab]")
{
    // Chart-unification step 2 (2026-06-04): can SME use the same
    // connected-metric-ball extractor as LME, with the C4 k-ring's
    // flatness rationale preserved by a tight radius instead of hop
    // counting? Sweeps Params::sme_chart_radius_mult over the two
    // cheap modal gates; the static heavyweights (Scordelis-Lo,
    // geodesic hemisphere, cylinder aspect ladder) are run separately
    // before any default change. mult=0 is the legacy k-ring
    // baseline.
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const auto mat = steel_033();

    // Gate 1: 32x8 polar disk, Leissa free-edge n=2 doublet.
    constexpr double R = 0.10;
    constexpr double h = 1.0e-3;
    const auto sm_disk = chladni::shell::shell_material_from_isotropic(mat, h);
    const auto disk = chladni::mesh::generate_circular_disk(R, 32, 8);
    const auto leissa =
        chladni::analytical::free_edge_circular_plate_angular_frequencies(
            {.radius = R, .thickness = h}, mat, 2);
    const double f_n2_ref = leissa[0] / two_pi;

    // Gate 2: icosphere k=2, Wilkinson n=2 spheroidal pentet.
    const auto sphere = chladni::mesh::generate_icosphere(R, 2);
    const auto sm_sph = chladni::shell::shell_material_from_isotropic(mat, h);
    const double w_wilkinson = chladni::analytical::
        complete_spherical_shell_wilkinson_angular_frequencies(
            {.radius = R, .thickness = h}, mat, 1).front();

    std::cout.setf(std::ios::unitbuf);
    std::cout << "[sme_chart_ab] mult=0 is the legacy k-ring-3 chart\n";
    for (double mult : {0.0, 2.5, 3.0, 3.5}) {
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme   = true;
        p.sme_chart_radius_mult  = mult;
        std::cout << "  mult=" << mult << ":";
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                disk.V, disk.F, mat, sm_disk, h, 2, asm_);
            const double f_n2 =
                0.5 * (modes.omegas(0) + modes.omegas(1)) / two_pi;
            std::cout << "  disk n=2 rel_err "
                      << (f_n2 - f_n2_ref) / f_n2_ref;
        } catch (const std::exception& e) {
            std::cout << "  disk FAILED (" << e.what() << ")";
        }
        try {
            chladni::shell::LMEAssembler asm_(p);
            const auto modes = chladni::shell::compute_shell_modes(
                sphere.V, sphere.F, mat, sm_sph, h, 5, asm_);
            double mean = 0.0;
            for (int i = 0; i < 5; ++i) mean += modes.omegas(i);
            mean /= 5.0;
            std::cout << "  icosphere pentet rel_err "
                      << (mean - w_wilkinson) / w_wilkinson;
        } catch (const std::exception& e) {
            std::cout << "  icosphere FAILED (" << e.what() << ")";
        }
        std::cout << '\n';
    }
}

TEST_CASE("LME geometric chart on a folded ribbon — no cross-sheet "
          "coupling away from the bend",
          "[lme][chart][connectivity][curved][ghost]")
{
    const FoldedRibbon rib;
    const auto mat = steel_033();
    const auto sm  = chladni::shell::shell_material_from_isotropic(mat, 0.01);

    LMEAssembler::Params p;  // defaults: curved + ghost, 1st-order LME
    LMEAssembler asm_(p);
    const Eigen::SparseMatrix<double> K =
        asm_.assemble_K(rib.V, rib.F, sm);
    const Eigen::SparseMatrix<double> M =
        asm_.assemble_M(rib.V, rib.F, mat.density * 0.01);

    // DISCRIMINATING cross-sheet pairs: mid-sheet (x <= 5) on one
    // side, near-bend (x' >= 10) on the other, both directions. The
    // assembly is per-patch (K(a,b) needs ONE chart holding both), so:
    //
    //  - WITHOUT the prune, mid-sheet charts anchored at x ≈ 9 hold
    //    the deep nodes (down to x ≈ 5) AND the opposite rim
    //    (x' >= 12, admitted by the 8-hop guard, Euclid-near at
    //    z-gap 0.5) as a DISCONNECTED island — the bridge at 15.5
    //    sits outside their ~4.7 ball. These pairs are nonzero.
    //
    //  - WITH the prune those charts are sheet-pure and no chart
    //    holds both: charts wide enough to span the bend connectedly
    //    (the x=13 rim anchors, whose h_a the 2.5 bridge edges
    //    inflate to ball ≈ 7.5) reach only down to x ≈ 6 — the
    //    assertion zones are out of reach, so the blocks are
    //    STRUCTURALLY zero.
    //
    // Pairs between the two bend-adjacent regions (x, x' >= 6) are
    // NOT asserted: the wide rim charts couple them legitimately —
    // the bridge lies inside those balls, keeping the set connected
    // (a genuine, if coarse, geodesic neighbourhood — exactly the
    // h_a-adaptive behaviour the paper intends).
    double cross_K = 0.0, cross_M = 0.0;
    int arg_x = -1, arg_y = -1, arg_xb = -1, arg_yb = -1;
    for (int xa = 0; xa < FoldedRibbon::nx; ++xa)
        for (int xb = 0; xb < FoldedRibbon::nx; ++xb) {
            const bool far_pair = (xa <= 5 && xb >= 10)
                               || (xa >= 10 && xb <= 5);
            if (!far_pair) continue;
            for (int y = 0; y < FoldedRibbon::ny; ++y)
                for (int yb = 0; yb < FoldedRibbon::ny; ++yb) {
                    const int i = FoldedRibbon::iA(xa, y);
                    const int j = FoldedRibbon::iB(xb, yb);
                    const double bk = block_max_abs(K, i, j);
                    if (bk > cross_K) {
                        cross_K = bk;
                        arg_x = xa; arg_y = y; arg_xb = xb; arg_yb = yb;
                    }
                    cross_M = std::max(cross_M, block_max_abs(M, i, j));
                }
        }
    INFO("max |K| over far cross-sheet blocks = " << cross_K
         << " at A(" << arg_x << "," << arg_y << ") x B("
         << arg_xb << "," << arg_yb << ")");
    INFO("max |M| over far cross-sheet blocks = " << cross_M);
    REQUIRE(cross_K == 0.0);
    REQUIRE(cross_M == 0.0);

    // The LEGITIMATE bend coupling must survive: charts anchored near
    // the fold contain the bridge inside their ball, so the opposite
    // rim is geodesically connected within the chart and stays.
    double bend_K = 0.0;
    for (int y = 0; y < FoldedRibbon::ny; ++y)
        for (int yb = 0; yb < FoldedRibbon::ny; ++yb)
            bend_K = std::max(
                bend_K,
                block_max_abs(K, FoldedRibbon::iA(13, y),
                              FoldedRibbon::iB(13, yb)));
    INFO("max |K| over A(13,·) x B(13,·) bend blocks = " << bend_K);
    REQUIRE(bend_K > 0.0);
}
