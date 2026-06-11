/**
 * @file test_sme_hemi_corner_dissection.cpp
 * @brief Dissection of ONE real failing SME (chart, Gauss point)
 *        instance from the geodesic pinched hemisphere — the
 *        pinch-corner marginal-infeasibility arc (faithful-first,
 *        2026-06-07). Sibling of @ref
 *        test_sme_rim_feasibility_dissection.cpp (cylinder rim), same
 *        method: embed the [sme_fail_dump] instance verbatim, certify
 *        the obstruction with a min-norm point, probe cures by
 *        construction.
 *
 * Context: with per-patch α escalation OFF, SME assembly at the
 * paper's α=2 THROWS on the coarse (142 V) geodesic quarter
 * hemisphere ([alpha_escalation] pins it). This is the last client of
 * the B3 escalation ladder after interior ownership (6214ed0) zeroed
 * the cylinder's Newton-drop channel. Unlike the cylinder failures
 * (chart-fringe queries, w_A ≈ 2e-4), these queries are CENTRAL:
 * the embedded instance fails at w_A = 0.866, essentially ON the
 * anchor node.
 *
 * The [hemi_corner_cls] diag (test_static_obstacle_course.cpp) shows
 * WHY: the failing charts contain a cluster of THREE adjacent d=0
 * (BoundaryCorner) nodes — gid 2, the GENUINE hole/symmetry-plane
 * pinch corner, flanked by gid 0 and gid 1, hole-rim nodes that are
 * SPURIOUSLY corner-classified because the face-dropped hole cut
 * leaves a staircase polyline whose turn angles (47–70°) exceed the
 * 45° corner threshold at every resolution (the spurious-corner count
 * GROWS with refinement: 4 at n=16 → 12 at n=64). The paper's d=0
 * corner recipe (§3.2.2) is being applied to discretization artefacts
 * of a SMOOTH boundary curve.
 *
 * Questions answered by construction:
 *  1. Does the embedded instance reproduce the failure standalone?
 *  2. Is it GENUINE infeasibility (min-norm point of conv{φ_a} has
 *     |g| > 0 ⇒ separating hyperplane), and what separates — the
 *     λ-block (1st-order, outside the hull) or the μ-block
 *     (2nd-order moment-with-gap)?
 *  3. WHAT CURES it:
 *      (a) the B3 ladder (all gaps ×2 / ×4) — the shipped rescue;
 *      (b) reclassifying ONLY the two spurious staircase corners
 *          (gid 0, 1) with the recipe's own flanking-node gap
 *          β h_t² t⊗t (BoundaryEdgeNearCorner), keeping the genuine
 *          corner at d=0 — if this restores feasibility, the root
 *          cause is the SPURIOUS CLASSIFICATION, not the paper's
 *          corner recipe, and the faithful fix is classification-
 *          level (the d=0 rule is only meant for true domain
 *          corners);
 *      (c) the same with the smaller mid-edge gap (α/4) h_t² t⊗t —
 *          how much tangential slack is actually needed.
 *
 * Tagged [.diag] — run on demand:
 *   ./build/tests/chladni_tests "[sme_hemi_dissect]"
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <string>
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

/// [sme_fail_dump] anchor=2 (the genuine hole/symmetry pinch corner of
/// the 142 V geodesic quarter hemisphere, k-ring chart, 21 nodes incl.
/// 7 ghosts). Rim order through the corner: 3 — 2 — 0 — 1 — 26.
/// k=0 (gid 0) and k=1 (gid 1) are the SPURIOUS staircase corners
/// (d=0); k=2 (gid 2) is the genuine pinch corner (d=0, anchor); k=3
/// (gid 3) and k=8 (gid 26) are their correctly-classified
/// BoundaryEdgeNearCorner flanks (rank-1 tangential gaps).
constexpr DumpNode kNodes[] = {
    {0,   false, -0.17800459968023546, 0.95093470494128096, 0.0, 0.0, 0.0},
    {1,   false, -0.74119294949162606, 0.083913278333764885, 0.0, 0.0, 0.0},
    {2,   false, 0.12957246613542853, -0.11723989030293615, 0.0, 0.0, 0.0},
    {3,   false, -0.40481311515381629, -1.018503595776084, 0.33239721406811118, 0.93573928724610567, 0.55770703077394101},
    {4,   false, -1.3587607850435206, -0.87677700581797935, 1.6372262061847127, 0.38652958177439251, -0.17849725864926491},
    {5,   false, -0.98357263396736805, -1.9845163301275233, 0.18098093719834965, 0.49851890536342203, 0.30037046909402976},
    {6,   false, -2.0115269676366614, -1.9040875386709177, 1.0142149980937196, 0.85302463983742627, -0.1447892796122574},
    {7,   false, -1.5877622301691208, -2.9814935256039172, 0.18418740104723424, 0.49531171253803247, 0.30204333636191177},
    {26,  false, -1.0865550263434884, 1.2321743199782387, 0.11762948869236156, 1.3201426714164857, -0.39406561311498556},
    {27,  false, -1.7426325304145871, 0.31446153573632535, 1.8977743070650031, 0.4504052161479411, -0.20430288025255217},
    {28,  false, -2.4459529995836213, -0.70440033007223657, 0.72208694105592974, 0.81335064878903818, 0.00026662370761725658},
    {38,  false, -1.4259045176030756, 2.3778664684643362, 0.11347242917829191, 1.314289396574954, -0.38618080023821932},
    {39,  false, -2.1161335378674009, 1.5387682246146266, 0.98426733030914348, 0.84113742604857666, 0.061269215029921387},
    {40,  false, -2.8669524398476951, 0.57225765556194452, 0.71119819676862395, 0.8605462056992963, -0.010961221750164271},
    {142, true,  -1.2707694539986738, 1.0435800370431179, 0.57365974713383372, 0.57365974713383372, 0.0},
    {143, true,  0.72659866304105403, 0.63226715489916718, 0.53082185779129321, 0.53082185779129321, 0.0},
    {144, true,  -0.051494858625523235, 0.92232984256536832, 0.61156828508443417, 0.61156828508443417, 0.0},
    {145, true,  0.63014867190543788, -1.0227422088852798, 0.52912226672740736, 0.52912226672740736, 0.0},
    {146, true,  0.097962670445439881, -1.9741216927693017, 0.60404595026835495, 0.60404595026835495, 0.0},
    {147, true,  -0.48093050066270787, -2.9663249000556018, 0.65805299777239279, 0.65805299777239279, 0.0},
    {159, true,  -0.39632600607916352, 2.0712725638279483, 0.65360473727125068, 0.65360473727125068, 0.0},
};
constexpr int kN = static_cast<int>(sizeof(kNodes) / sizeof(kNodes[0]));

// Embedded-instance indices of the d=0 cluster (k into kNodes).
constexpr int kSpur0  = 0;  ///< gid 0 — spurious staircase corner
constexpr int kSpur1  = 1;  ///< gid 1 — spurious staircase corner
constexpr int kCorner = 2;  ///< gid 2 — genuine pinch corner (anchor)
constexpr int kFlank3 = 3;  ///< gid 3 — EdgeNearCorner flank (sym edge)
constexpr int kFlank26 = 8; ///< gid 26 — EdgeNearCorner flank (rim)

constexpr double kQueryX0 = 0.055232000247736272;
constexpr double kQueryX1 = -0.037159252612987213;
constexpr double kHChart  = 0.98703939140020513;
// Assembler r_cut at the shipped α=2, TOL_LME=1e-10, in SCALED units
// (h_chart = 1) — same formula as the cylinder dissection.
const double kRCutScaled = std::sqrt(-std::log(1e-10) * 2.0 / 2.0);

/// Scaled (h_chart=1) instance, exactly as the assembler dispatch
/// builds it, over a caller-supplied node list with optional uniform
/// gap scale (the B3 α-escalation probe).
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

/// Assign node @p k the rank-1 tangential gap c · h_t² t⊗t the §3.2.2
/// recipe would produce had it NOT been corner-classified: t along the
/// rim chord between its two rim neighbours, h_t the true directional
/// spacing max over those neighbours (S-D1's definition).
void regap_tangential(std::vector<DumpNode>& list, int k,
                      int nbr_a, int nbr_b, double coeff)
{
    const Eigen::Vector2d xa(list[static_cast<std::size_t>(k)].x0,
                             list[static_cast<std::size_t>(k)].x1);
    const Eigen::Vector2d pa(list[static_cast<std::size_t>(nbr_a)].x0,
                             list[static_cast<std::size_t>(nbr_a)].x1);
    const Eigen::Vector2d pb(list[static_cast<std::size_t>(nbr_b)].x0,
                             list[static_cast<std::size_t>(nbr_b)].x1);
    const Eigen::Vector2d t = (pb - pa).normalized();
    const double ht = std::max(std::abs((pa - xa).dot(t)),
                               std::abs((pb - xa).dot(t)));
    const Eigen::Matrix2d d = coeff * ht * ht * (t * t.transpose());
    list[static_cast<std::size_t>(k)].d11 = d(0, 0);
    list[static_cast<std::size_t>(k)].d22 = d(1, 1);
    list[static_cast<std::size_t>(k)].d12 = d(0, 1);
}

/// Min-norm point of conv{φ_a} via Gilbert's algorithm — identical to
/// the cylinder dissection. φ_a = (u_a1, u_a2, −D11, −D22, −2 D12)
/// with u_a = x − x_a, D_a = (x_a−x)(x_a−x)ᵀ − d_a. Feasible iff
/// 0 ∈ int conv{φ_a}; |g| > 0 certifies a separating direction
/// v = −g/|g| with v·φ_a ≤ −|g| < 0 ∀a (up to solver gap).
struct MinNorm {
    Eigen::VectorXd g;
    double          norm;
    Eigen::VectorXd margins;
};

MinNorm min_norm_point(const ScaledInstance& s)
{
    const auto n = s.nodes.rows();
    Eigen::MatrixXd PHI(n, 5);
    for (Eigen::Index a = 0; a < n; ++a) {
        const Eigen::Vector2d u = s.x - s.nodes.row(a).transpose();
        const Eigen::Vector2d w = -u;
        const Eigen::Matrix2d D =
            w * w.transpose() - s.d[static_cast<std::size_t>(a)];
        PHI.row(a) << u(0), u(1), -D(0, 0), -D(1, 1), -2.0 * D(0, 1);
    }
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

/// Solve + min-norm report for one probe variant.
void probe(const char* label, const std::vector<DumpNode>& list,
           double gap_scale = 1.0)
{
    const ScaledInstance e = build_scaled(list, gap_scale);
    bool ok = true;
    std::string err;
    try {
        (void)chladni::shell::lme::evaluate_sme_basis(
            e.nodes, e.d, e.x, kRCutScaled, 1e-10, 125);
    } catch (const std::runtime_error& ex) {
        ok = false; err = ex.what();
    }
    const MinNorm m = min_norm_point(e);
    std::fprintf(stderr,
        "[sme_hemi_dissect] %-44s solve %s, min-norm |g| = %.3e\n",
        label, ok ? "CONVERGES" : "fails", m.norm);
}

}  // namespace

TEST_CASE("SME hemisphere pinch-corner infeasibility — dissect one real "
          "failing instance",
          "[.diag][shell][sme][sme_hemi_dissect]")
{
    using chladni::shell::lme::evaluate_sme_basis;

    // ---- 1. The embedded instance reproduces the failure. ----------
    const ScaledInstance inst = build_scaled(dump_nodes());
    REQUIRE_THROWS_AS(
        evaluate_sme_basis(inst.nodes, inst.d, inst.x, kRCutScaled,
                           1e-10, 125),
        std::runtime_error);
    std::fprintf(stderr,
        "[sme_hemi_dissect] embedded instance: evaluate_sme_basis "
        "THROWS at the assembler's parameters (reproduced)\n");

    // ---- 2. Separating direction. -----------------------------------
    const MinNorm mn = min_norm_point(inst);
    {
        const Eigen::VectorXd v = -mn.g / (mn.norm + 1e-300);
        Eigen::Matrix2d vmu;
        vmu << v(2), 0.5 * v(4), 0.5 * v(4), v(3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(vmu);
        std::fprintf(stderr,
            "[sme_hemi_dissect] min-norm |g| = %.3e (0 outside hull "
            "⇔ infeasible)\n"
            "[sme_hemi_dissect] v = (λ: %.3f %.3f | μ: %.3f %.3f %.3f)"
            "  |v_λ| = %.3f  |v_μ| = %.3f\n"
            "[sme_hemi_dissect] v_μ eigen: (%.3e along [%.3f %.3f]) "
            "(%.3e along [%.3f %.3f])\n"
            "[sme_hemi_dissect] worst margin max_a v·φ_a = %.3e "
            "(< 0 confirms separation)\n",
            mn.norm,
            v(0), v(1), v(2), v(3), v(4),
            v.head<2>().norm(), v.tail<3>().norm(),
            es.eigenvalues()(0), es.eigenvectors()(0, 0),
            es.eigenvectors()(1, 0),
            es.eigenvalues()(1), es.eigenvectors()(0, 1),
            es.eigenvectors()(1, 1),
            mn.margins.maxCoeff());
        // Near-active = margin within 2x of the best (max is < 0 for
        // a genuinely separated instance).
        for (int a = 0; a < kN; ++a) {
            if (mn.margins(a) >= 2.0 * mn.margins.maxCoeff()) {
                std::fprintf(stderr,
                    "[sme_hemi_dissect]   near-active node k=%d gid=%d"
                    " ghost=%d xi=(%.4f %.4f) d=0:%d margin=%.3e\n",
                    a, kNodes[a].gid, kNodes[a].ghost ? 1 : 0,
                    kNodes[a].x0, kNodes[a].x1,
                    (kNodes[a].d11 == 0.0 && kNodes[a].d22 == 0.0)
                        ? 1 : 0,
                    mn.margins(a));
            }
        }
        REQUIRE(mn.margins.maxCoeff() < 0.0);  // genuinely separated
    }

    // ---- 3a. B3 ladder probe (all gaps ×2, ×4). ---------------------
    probe("B3 ladder: all gaps x2", dump_nodes(), 2.0);
    probe("B3 ladder: all gaps x4", dump_nodes(), 4.0);

    // ---- 3b. Reclassification probes. -------------------------------
    // The counterfactual the [hemi_corner_cls] diag motivates: give the
    // two SPURIOUS staircase corners the tangential gap the recipe
    // gives nodes flanking a corner (BoundaryEdgeNearCorner,
    // β h_t² t⊗t, β=1), keeping the GENUINE pinch corner at d=0.
    // Rim order 3 — 2 — 0 — 1 — 26 fixes each node's rim neighbours.
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kSpur0, kCorner, kSpur1, 1.0);   // gid 0
        regap_tangential(re, kSpur1, kSpur0, kFlank26, 1.0);  // gid 1
        probe("reclassify spurious corners as EdgeNC (b=1)", re);
    }
    // Same, with the smaller mid-edge coefficient (α/4 = 0.5 at α=2) —
    // how much tangential slack the obstruction actually needs.
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kSpur0, kCorner, kSpur1, 0.5);
        regap_tangential(re, kSpur1, kSpur0, kFlank26, 0.5);
        probe("reclassify spurious corners as EdgeMid (a/4)", re);
    }
    // Control: relax only ONE spurious corner — is the cluster the
    // obstruction, or a single d=0 neighbour?
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kSpur0, kCorner, kSpur1, 1.0);
        probe("reclassify gid 0 only (EdgeNC)", re);
    }
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kSpur1, kSpur0, kFlank26, 1.0);
        probe("reclassify gid 1 only (EdgeNC)", re);
    }
    // Control: genuine corner relaxed too (all three tangential) —
    // establishes whether the paper's d=0 at the TRUE corner is fine
    // once the spurious ones are gone.
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kSpur0, kCorner, kSpur1, 1.0);
        regap_tangential(re, kSpur1, kSpur0, kFlank26, 1.0);
        regap_tangential(re, kCorner, kFlank3, kSpur0, 1.0);
        probe("ALSO relax the genuine corner (EdgeNC)", re);
    }
    // Decisive control: relax ONLY the genuine corner, keep both
    // spurious staircase corners at d=0 — is gid 2's d=0 the binding
    // constraint all by itself? (The query sits 0.11 h from gid 2.)
    {
        std::vector<DumpNode> re = dump_nodes();
        regap_tangential(re, kCorner, kFlank3, kSpur0, 1.0);
        probe("relax ONLY the genuine corner (EdgeNC)", re);
    }

    SUCCEED("dissection printed to stderr");
}
