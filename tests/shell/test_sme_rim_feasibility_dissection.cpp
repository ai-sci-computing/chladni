/**
 * @file test_sme_rim_feasibility_dissection.cpp
 * @brief Dissection of ONE real failing SME (chart, Gauss point)
 *        instance from the cylinder — the rim-infeasibility
 *        root-cause arc (faithful-first, 2026-06-07).
 *
 * Context: at the paper's own parameters (α=2, β=1), EVERY rim-region
 * SME anchor on the free-free cylinder fails a handful of Gauss-point
 * solves (fails=5–23 per anchor, both aspects, both chart
 * extractors), and the SME_DIAG recession-ray test classifies 98.6 %
 * of them as GENUINELY INFEASIBLE (separating hyperplane — 0 outside
 * conv{φ_a}), not Newton conditioning. These failures are what the B1
 * drop-net currently absorbs — the load-bearing invention the
 * faithful-first push wants to retire.
 *
 * This test embeds one such instance verbatim (LME_DIAG
 * [sme_fail_dump], anchor 3 of the 24x32 aspect-2.39 cylinder,
 * k-ring-3 chart, 28 nodes incl. 6 ghosts; the failing query is a
 * 12-pt-rule Gauss point at the chart's TANGENTIAL fringe, Shepard
 * weight w_A ≈ 2e-4) and answers, by construction rather than
 * conjecture:
 *
 *  1. Is the embedded instance infeasible through the public API
 *     (reproduces the assembler failure standalone)?
 *  2. WHAT separates — the min-norm point of conv{φ_a}
 *     (Gilbert/Frank-Wolfe) gives the separating direction v ∈ R^5;
 *     its λ/μ-block decomposition says whether the obstruction is
 *     1st-order (outside the hull — would implicate ownership) or
 *     2nd-order (moment-with-gap — implicates the node-set/gap
 *     design), and the μ eigenvector names the offending direction.
 *  3. WHAT CURES it:
 *      (a) α-escalation ×4 on the patch gaps — the assembler's B3
 *          ladder (measured: rescues almost nothing; expect FAIL);
 *      (b) extending the node set with the mesh's own circumferential
 *          continuation (the lattice rows the k-ring chart CUT OFF) —
 *          if this restores feasibility, the root cause is the
 *          IN-CHART RESTRICTION of the paper's global cloud, not the
 *          §3.2.2 gap recipe: the paper guarantees feasibility for
 *          the full cloud; our chart truncation voids the guarantee
 *          exactly at chart-fringe queries, where coverage becomes
 *          one-sided. (1st-order LME does not suffer because its
 *          feasibility condition is only x ∈ conv(nodes), which
 *          quadrature ownership DOES guarantee.)
 *
 * Tagged [.diag] — run on demand:
 *   ./build/tests/chladni_tests "[sme_rim_dissect]"
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

/// One chart node of the embedded failing instance (chart-PHYSICAL
/// coordinates, exactly as dumped by [sme_fail_dump]).
struct DumpNode {
    int    gid;
    bool   ghost;
    double x0, x1;        ///< chart position
    double d11, d22, d12; ///< gap matrix entries
};

/// [sme_fail_dump] anchor=3 (rim anchor, 24x32 aspect-2.39 cylinder,
/// k-ring-3 chart): 7 rim nodes (tangential-only gaps), 6+5+4
/// interior rows (β h_n² normal + tangential slack on row +1,
/// anisotropic interior beyond), 6 ghosts (isotropic E3 gaps).
/// Chart x ≈ rim normal (axial), chart y ≈ rim tangent
/// (circumferential).
constexpr DumpNode kNodes[] = {
    {0,   false, 0.00060070830695526766, -0.070709737809878492, 1.9989030179601437e-09, 0.0002244384425495034, -6.6979898488910586e-07},
    {1,   false, 0.00050066996444161948, -0.050020471608403914, 1.1978734435052873e-08, 0.00029747063089049581, -1.8876762671774216e-06},
    {2,   false, 0.0003164613756398308, -0.025916509512569436, 2.6219650516994252e-08, 0.00033825690376038293, -2.9780829070994314e-06},
    {3,   false, 6.063605145765699e-05, -4.0496705525438845e-05, 4.0037277351457087e-08, 0.00033503275883137797, -3.6624854248386729e-06},
    {4,   false, -0.00024937193503322416, 0.025844159302061059, 5.7119229679950815e-08, 0.00033499824994793218, -4.3743390336321113e-06},
    {5,   false, -0.00059243605186588019, 0.049973461979689639, 7.0035866448271008e-08, 0.00029109994772394306, -4.5152449614492404e-06},
    {6,   false, -0.00094517704641842375, 0.070703039225437955, 7.641120991681205e-08, 0.00021484275901687192, -4.0517150884958914e-06},
    {25,  false, 0.062994960278529161, -0.04933662908341941, 0.0039213501509783397, 0.00030025029943309307, 4.1459695919611682e-05},
    {26,  false, 0.06281075168972737, -0.025232666987584935, 0.0039155039828249688, 0.00034062218716474351, 4.0245568657119364e-05},
    {27,  false, 0.062554926365545202, 0.00064334581945906162, 0.0039089845647233877, 0.0003371997120109368, 3.9444500241523766e-05},
    {28,  false, 0.062244918379054319, 0.026528001827045559, 0.0039055843743373, 0.00033547534891451881, 3.872402824053957e-05},
    {29,  false, 0.06190185426222166, 0.050657304504674143, 0.0039055951683555851, 0.00029157926445328412, 3.8597304329055862e-05},
    {30,  false, 0.061549113267669119, 0.071386881750422446, 0.0039055985349660751, 0.00021532532699270005, 3.9080167438340765e-05},
    {50,  false, 0.1253050420038149, -0.024548824462600434, 0.0021221170342316195, 0.0004201834964399243, 0.00042849445103731781},
    {51,  false, 0.12504921667963273, 0.001327188344443562, 0.0021204346209912493, 0.00042804823308460727, 0.00044637195165737377},
    {52,  false, 0.12473920869314184, 0.027211844352030057, 0.0021189532984171185, 0.00042045593131986999, 0.00042749728197130908},
    {53,  false, 0.12439614457630919, 0.05134114702965864, 0.0020930818852855112, 0.00035991293227111024, 0.00037923898232605552},
    {54,  false, 0.12404340358175664, 0.07207072427540695, 0.0020495577976602378, 0.00026187428865348267, 0.00030836462108144001},
    {75,  false, 0.18754350699372027, 0.0020110308694280621, 0.0021204346209912493, 0.00042804823308460727, 0.00044637195165737377},
    {76,  false, 0.18723349900722938, 0.027895686877014558, 0.0021189532984171185, 0.00042045593131986999, 0.00042749728197130908},
    {77,  false, 0.18689043489039672, 0.052024989554643145, 0.0020930818852855112, 0.00035991293227111024, 0.00037923898232605552},
    {78,  false, 0.18653769389584418, 0.072754566800391454, 0.0020495577976602378, 0.00026187428865348267, 0.00030836462108144001},
    {792, true,  -0.061943601178389096, -0.061048947234125697, 0.0013578658452900697, 0.0013578658452900697, 0.0},
    {794, true,  -0.062085724644046818, -0.038652333085471169, 0.0013578658452900697, 0.0013578658452900697, 0.0},
    {795, true,  -0.062305741600538797, -0.013662345634031941, 0.0013578658452900697, 0.0013578658452900697, 0.0},
    {796, true,  -0.062588658255875326, 0.012217988773283308, 0.0013578658452900697, 0.0013578658452900697, 0.0},
    {797, true,  -0.062915194307537087, 0.03722496811589085, 0.0013578658452900697, 0.0013578658452900697, 0.0},
    {798, true,  -0.063263096863229687, 0.059654408077579296, 0.0013578658452900697, 0.0013578658452900697, 0.0},
};
constexpr int kN = static_cast<int>(sizeof(kNodes) / sizeof(kNodes[0]));

constexpr double kQueryX0  = 0.004530788853335771;
constexpr double kQueryX1  = -0.068056064028503102;
constexpr double kHChart   = 0.052112682626978046;
// Assembler r_cut at the shipped α=2, TOL_LME=1e-10:
// r = h_max·sqrt(ln(1e10)·α/2) ≈ 4.7985·h_max; h_max ≈ h_chart on
// this near-uniform grid. In SCALED units (h_chart = 1):
const double kRCutScaled = std::sqrt(-std::log(1e-10) * 2.0 / 2.0);

/// Build the SCALED (h_chart=1) node matrix / gaps / query exactly as
/// the assembler dispatch does, with an optional per-node gap scale
/// (the α-escalation probe) over a caller-supplied node list.
struct ScaledInstance {
    Eigen::MatrixXd              nodes;
    std::vector<Eigen::Matrix2d> d;
    Eigen::Vector2d              x;
};

ScaledInstance build_scaled(const std::vector<DumpNode>& list,
                            double gap_scale = 1.0)
{
    ScaledInstance s;
    const auto n = static_cast<Eigen::Index>(list.size());
    s.nodes.resize(n, 2);
    s.d.reserve(list.size());
    const double inv_h  = 1.0 / kHChart;
    const double inv_h2 = inv_h * inv_h;
    for (Eigen::Index k = 0; k < n; ++k) {
        const DumpNode& nd = list[static_cast<std::size_t>(k)];
        s.nodes(k, 0) = nd.x0 * inv_h;
        s.nodes(k, 1) = nd.x1 * inv_h;
        Eigen::Matrix2d dm;
        dm << nd.d11, nd.d12, nd.d12, nd.d22;
        s.d.push_back(dm * inv_h2 * gap_scale);
    }
    s.x << kQueryX0 * inv_h, kQueryX1 * inv_h;
    return s;
}

std::vector<DumpNode> dump_nodes()
{
    return {kNodes, kNodes + kN};
}

/// Min-norm point of conv{φ_a} via Gilbert's algorithm. φ_a is the
/// SME dual-constraint vector (u_a1, u_a2, −D11, −D22, −2 D12) with
/// u_a = x − x_a and D_a = (x_a−x)(x_a−x)ᵀ − d_a. The SME program at
/// x is feasible iff 0 ∈ int conv{φ_a}; a min-norm point g with
/// |g| > 0 yields the separating direction v = −g/|g|
/// (v·φ_a ≤ −|g| < 0 ∀a up to the solver gap).
struct MinNorm {
    Eigen::VectorXd g;          ///< min-norm point (5-vector)
    double          norm;       ///< |g|
    Eigen::VectorXd margins;    ///< v·φ_a per node, v = −g/|g|
};

MinNorm min_norm_point(const ScaledInstance& s)
{
    const auto n = s.nodes.rows();
    Eigen::MatrixXd PHI(n, 5);
    for (Eigen::Index a = 0; a < n; ++a) {
        const Eigen::Vector2d u = s.x - s.nodes.row(a).transpose();
        const Eigen::Vector2d w = -u;  // x_a − x
        const Eigen::Matrix2d D =
            w * w.transpose() - s.d[static_cast<std::size_t>(a)];
        PHI.row(a) << u(0), u(1), -D(0, 0), -D(1, 1), -2.0 * D(0, 1);
    }
    // Gilbert iteration. Sublinear, so run LONG (n=28, 5-D — cheap);
    // stop when the Frank-Wolfe duality gap g·g − min_a g·φ_a (≥ 0,
    // = 0 at the min-norm point) is negligible relative to |g|².
    Eigen::Index a0 = 0;
    PHI.rowwise().norm().minCoeff(&a0);
    Eigen::VectorXd g = PHI.row(a0).transpose();
    for (int it = 0; it < 2000000; ++it) {
        Eigen::Index s_idx = 0;
        const double gphi_min = (PHI * g).minCoeff(&s_idx);
        if (g.squaredNorm() - gphi_min
            < 1e-10 * std::max(1.0, g.squaredNorm()))
            break;
        const Eigen::VectorXd phi_s = PHI.row(s_idx).transpose();
        const Eigen::VectorXd diff  = g - phi_s;
        const double denom = diff.squaredNorm();
        if (denom < 1e-30) break;
        const double gamma =
            std::min(1.0, std::max(0.0, g.dot(diff) / denom));
        if (gamma <= 0.0) break;
        g = (1.0 - gamma) * g + gamma * phi_s;
    }
    MinNorm out;
    out.g    = g;
    out.norm = g.norm();
    const Eigen::VectorXd v = -g / (out.norm + 1e-300);
    out.margins = PHI * v;
    return out;
}

}  // namespace

TEST_CASE("SME rim infeasibility — dissect one real failing instance",
          "[.diag][shell][sme][sme_rim_dissect]")
{
    using chladni::shell::lme::evaluate_sme_basis;

    // ---- 1. The embedded instance reproduces the failure. ----------
    const ScaledInstance inst = build_scaled(dump_nodes());
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(inst.nodes, inst.d, inst.x, kRCutScaled,
                           1e-10, 125),
        std::runtime_error);
    std::fprintf(stderr,
        "[sme_rim_dissect] embedded instance: evaluate_sme_basis "
        "THROWS at the assembler's parameters (reproduced)\n");

    // ---- 2. Separating direction. -----------------------------------
    const MinNorm mn = min_norm_point(inst);
    {
        Eigen::Matrix2d vmu;
        const Eigen::VectorXd v = -mn.g / (mn.norm + 1e-300);
        vmu << v(2), 0.5 * v(4), 0.5 * v(4), v(3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(vmu);
        std::fprintf(stderr,
            "[sme_rim_dissect] min-norm |g| = %.3e (0 outside hull "
            "⇔ infeasible)\n"
            "[sme_rim_dissect] v = (λ: %.3f %.3f | μ: %.3f %.3f %.3f)"
            "  |v_λ| = %.3f  |v_μ| = %.3f\n"
            "[sme_rim_dissect] v_μ eigen: (%.3e along [%.3f %.3f]) "
            "(%.3e along [%.3f %.3f])\n"
            "[sme_rim_dissect] worst margin max_a v·φ_a = %.3e "
            "(< 0 confirms separation)\n",
            mn.norm,
            v(0), v(1), v(2), v(3), v(4),
            v.head<2>().norm(), v.tail<3>().norm(),
            es.eigenvalues()(0), es.eigenvectors()(0, 0),
            es.eigenvectors()(1, 0),
            es.eigenvalues()(1), es.eigenvectors()(0, 1),
            es.eigenvectors()(1, 1),
            mn.margins.maxCoeff());
        // Report which nodes sit ON the obstruction (margin nearest 0).
        for (int a = 0; a < kN; ++a) {
            if (mn.margins(a) > 0.5 * mn.margins.maxCoeff()) {
                std::fprintf(stderr,
                    "[sme_rim_dissect]   near-active node k=%d gid=%d"
                    " ghost=%d xi=(%.4f %.4f) margin=%.3e\n",
                    a, kNodes[a].gid, kNodes[a].ghost ? 1 : 0,
                    kNodes[a].x0, kNodes[a].x1, mn.margins(a));
            }
        }
        REQUIRE(mn.margins.maxCoeff() < 0.0);  // genuinely separated
    }

    // ---- 3a. α-escalation probe (B3 ladder: gaps ×2, ×4). -----------
    for (double esc : {2.0, 4.0}) {
        const ScaledInstance e = build_scaled(dump_nodes(), esc);
        bool ok = true;
        try {
            (void)evaluate_sme_basis(e.nodes, e.d, e.x, kRCutScaled,
                                     1e-10, 125);
        } catch (const std::runtime_error&) { ok = false; }
        const MinNorm m = min_norm_point(e);
        std::fprintf(stderr,
            "[sme_rim_dissect] gaps ×%.0f: solve %s, min-norm |g| ="
            " %.3e\n",
            esc, ok ? "CONVERGES" : "fails", m.norm);
    }

    // ---- 3b. Lattice-continuation probe: restore the rows the ------
    //          k-ring chart cut off in −y (the mesh's circumferential
    //          continuation; positions extrapolated from the uniform
    //          lattice step of each row, gaps copied from the row's
    //          existing nodes). If THIS restores feasibility, the
    //          obstruction is the chart's one-sided coverage at the
    //          fringe — the paper's global-cloud feasibility guarantee
    //          voided by the in-chart restriction — NOT the gap
    //          recipe.
    {
        std::vector<DumpNode> ext = dump_nodes();
        // Row layout in the dump: [0..6] rim, [7..12] row+1,
        // [13..17] row+2, [18..21] row+3, [22..27] ghosts. Extend each
        // by TWO nodes continuing in −y with the row's own step.
        const int row_start[5] = {0, 7, 13, 18, 22};
        for (int n_add : {1, 2}) {
            std::vector<DumpNode> ext2 = ext;
            for (int r = 0; r < 5; ++r) {
                const DumpNode& n0 = kNodes[row_start[r]];
                const DumpNode& n1 = kNodes[row_start[r] + 1];
                const double dy = n1.x1 - n0.x1;   // lattice step (+y)
                const double dx = n1.x0 - n0.x0;
                for (int j = 1; j <= n_add; ++j) {
                    DumpNode add = n0;             // copy gid/ghost/gaps
                    add.gid = -1;                  // synthetic
                    add.x0  = n0.x0 - j * dx;
                    add.x1  = n0.x1 - j * dy;
                    ext2.push_back(add);
                }
            }
            const ScaledInstance e = build_scaled(ext2);
            bool ok = true;
            std::string err;
            try {
                (void)evaluate_sme_basis(e.nodes, e.d, e.x,
                                         kRCutScaled, 1e-10, 125);
            } catch (const std::runtime_error& ex) {
                ok = false; err = ex.what();
            }
            const MinNorm m = min_norm_point(e);
            std::fprintf(stderr,
                "[sme_rim_dissect] lattice continuation (+%d node%s/"
                "row in −y): solve %s, min-norm |g| = %.3e%s%s\n",
                n_add, n_add > 1 ? "s" : "",
                ok ? "CONVERGES" : "fails", m.norm,
                ok ? "" : " — ", ok ? "" : err.c_str());
        }
    }

    SUCCEED("dissection printed to stderr");
}
