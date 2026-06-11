/**
 * @file test_lme_sme_chart_diag.cpp
 * @brief Diagnostic: SME degree-2 polynomial reproduction on REAL
 *        wPCA-projected charts from production meshes.
 *
 * The 5x5 paper-Fig.-6 grid in @ref test_lme_sme_basis.cpp confirms
 * the SME basis reproduces affine and quadratic polynomials to
 * Newton tolerance on a synthetic regular grid. That fixture does
 * NOT exercise:
 *
 *  - **wPCA-projected anisotropic charts** (the icosphere chart's
 *    xi positions sit on a slightly-curved manifold projection,
 *    not a uniform grid).
 *  - **k-ring-3 neighbour sets** (varying chart size, ~26 nodes on
 *    icosphere k=1, ~37 on polar disk 32x8).
 *  - **The chart-2D outer-30 %-radius BoundaryCorner classification**
 *    the assembler shipped in the 2026-05 era. ⚠️RETIRED 2026-06-02:
 *    production now classifies from GLOBAL boundary frames with the
 *    paper §3.2.2 graded recipe (compute_sme_boundary_frames) — no
 *    chart-outer-radius rule and no blanket boundary→d=0 exist in the
 *    assembler any more (and 2026-06-06 measurement shows the live
 *    classification is chart-shape-INSENSITIVE; see inventory C4/D).
 *    The Classifier::ShippedB1b harness below is preserved as ERA
 *    HISTORY — it documents what this diag originally compared.
 *  - **The xi/h chart renormalisation** the assembler applies before
 *    each SME call.
 *
 * This test extracts one chart from each of two production meshes —
 * icosphere k=2 and polar disk 32x8 — replicates the assembler's
 * chart-construction logic exactly, and checks whether SME still
 * reproduces a degree-2 polynomial at chart-interior query points.
 * (The motivating numbers at write-time, 2026-05-23 — "SME regresses
 * the icosphere 2.25 % → 6.07 % vs LME, wins the disk 7.9 % → 0.11 %"
 * — are era history: they predate both the faithful-SME work of
 * 2026-06-02/03 and the per-node-β LME derivative fix `edaf0e1`.
 * Current readings on the disk: LME +0.75 %, SME +0.73 %.)
 *
 *  - If both charts reproduce cleanly, the SME basis evaluator is
 *    sound on real chart geometries and the icosphere regression
 *    lives ABOVE the basis (in the strain-B assembly, the eigensolver,
 *    or the curvature-coupling pieces).
 *  - If the icosphere chart fails reproduction and the polar disk
 *    chart passes, the basis evaluator has a flaw the synthetic
 *    grid doesn't surface — likely a chart-anisotropy or
 *    classification-mismatch issue.
 *
 * Tagged @c [.diag] — runs on demand, not in the default suite.
 */

#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <queue>
#include <vector>

using chladni::shell::LMEAssembler;
using chladni::shell::lme::NodalGap;
using chladni::shell::lme::NodalGapKind;
using chladni::shell::lme::Patch;
using chladni::shell::lme::build_patch;
using chladni::shell::lme::compute_nodal_gaps;
using chladni::shell::lme::evaluate_sme_basis;
using chladni::shell::lme::evaluate_sme_basis_and_grad;
using chladni::shell::lme::evaluate_sme_basis_grad_and_hess;
using chladni::shell::lme::evaluate_sme_basis_grad_and_hess_closed_form;
using chladni::shell::lme::evaluate_basis;
using chladni::shell::lme::evaluate_basis_grad_and_hess;

namespace {

/// Replica of @c build_vertex_adjacency from @c src/shell/lme.cpp.
/// One sorted, deduped neighbour list per vertex from the F edge set.
std::vector<std::vector<int>> diag_vertex_adjacency(
    int n_v, const Eigen::MatrixXi& F)
{
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n_v));
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        const int a = F(t, 0), b = F(t, 1), c = F(t, 2);
        adj[static_cast<std::size_t>(a)].push_back(b);
        adj[static_cast<std::size_t>(a)].push_back(c);
        adj[static_cast<std::size_t>(b)].push_back(a);
        adj[static_cast<std::size_t>(b)].push_back(c);
        adj[static_cast<std::size_t>(c)].push_back(a);
        adj[static_cast<std::size_t>(c)].push_back(b);
    }
    for (auto& nbrs : adj) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return adj;
}

/// Replica of @c k_ring from @c src/shell/lme.cpp.
std::vector<int> diag_k_ring(
    int anchor, int k_max,
    const std::vector<std::vector<int>>& adj)
{
    const int n_v = static_cast<int>(adj.size());
    std::vector<int> depth(static_cast<std::size_t>(n_v), -1);
    depth[static_cast<std::size_t>(anchor)] = 0;
    std::queue<int> q;
    q.push(anchor);
    while (!q.empty()) {
        const int u = q.front(); q.pop();
        const int d = depth[static_cast<std::size_t>(u)];
        if (d >= k_max) continue;
        for (int v : adj[static_cast<std::size_t>(u)]) {
            if (depth[static_cast<std::size_t>(v)] == -1) {
                depth[static_cast<std::size_t>(v)] = d + 1;
                q.push(v);
            }
        }
    }
    std::vector<int> out;
    for (int v = 0; v < n_v; ++v) {
        if (depth[static_cast<std::size_t>(v)] != -1) out.push_back(v);
    }
    return out;
}

/// Approximate one-ring h per vertex (mean edge length to its
/// one-ring neighbours). Mirrors @c vertex_one_ring_h.
Eigen::VectorXd diag_vertex_h(
    const Eigen::MatrixXd& V,
    const std::vector<std::vector<int>>& adj)
{
    const Eigen::Index n_v = V.rows();
    Eigen::VectorXd h(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        const auto& nbrs = adj[static_cast<std::size_t>(a)];
        double s = 0.0;
        for (int b : nbrs) {
            s += (V.row(a) - V.row(b)).norm();
        }
        h(a) = nbrs.empty() ? 1.0 : s / static_cast<double>(nbrs.size());
    }
    return h;
}

/// MAX incident edge length per vertex — the paper's worst-case nodal
/// spacing for the interior gap d_a = (α/4) max(h²_{a-1}, h²_a)
/// (Rosolen-Millán-Arroyo 2013 §3.2.1, Eq. for d_a). Contrast with
/// @ref diag_vertex_h (MEAN), which the production code uses.
Eigen::VectorXd diag_vertex_h_max(
    const Eigen::MatrixXd& V,
    const std::vector<std::vector<int>>& adj)
{
    const Eigen::Index n_v = V.rows();
    Eigen::VectorXd h(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        const auto& nbrs = adj[static_cast<std::size_t>(a)];
        double m = 0.0;
        for (int b : nbrs) {
            m = std::max(m, (V.row(a) - V.row(b)).norm());
        }
        h(a) = nbrs.empty() ? 1.0 : m;
    }
    return h;
}

enum class HMeasure { Mean, Max };

struct PolyReproResult {
    int    anchor;
    int    n_chart;
    double h_chart;
    double xi_radius_max;
    int    n_boundary_corner;
    double max_err_const;        // PoU: Σ s_a = 1 (exact)
    double max_err_linear;       // 1st moment: Σ s_a (x_a-x) = 0 (exact)
    // SME guarantees Σ s_a (x_a-x)(x_a-x)^T = Σ s_a d_a;
    // measure the residual of this constraint directly.
    double max_residual_2nd_mom;
    // For comparison: what the "naive 2nd-order reproduction" error
    // looks like (this is EXPECTED to be O(d̄) and is NOT a bug).
    double max_err_naive_quad;
    // Magnitude of the d̄ slack at each query (tr(Σ s_a d_a))
    // — expected to bound max_err_naive_quad above.
    double max_d_slack_trace;
    int    n_queries_tested;
    int    n_queries_diverged;
    // Divergence split by query radius class: center (r=0),
    // inner (0.25·rmax), outer (0.50·rmax).
    int    n_div_center = 0;
    int    n_div_inner  = 0;
    int    n_div_outer  = 0;
};

/// Build the chart at @p anchor with the assembler's settings,
/// classify with the shipped rule (global-boundary +
/// chart-2D outer-30 %-radius → BoundaryCorner), renormalise to
/// h_chart=1, then check SME polynomial reproduction at a set of
/// chart-interior query points.
enum class Classifier {
    /// 2026-05-ERA shipped rule (RETIRED 2026-06-02, kept as history):
    /// global mesh boundary + chart-2D outer-30 %-radius
    /// → BoundaryCorner (d=0); else Interior. Production now uses the
    /// §3.2.2 graded global-boundary-frame recipe instead.
    ShippedB1b,
    /// All chart nodes Interior (d = (α/4) h² I everywhere) — tests
    /// whether the BoundaryCorner zeroing is what breaks degree-2
    /// reproduction.
    AllInterior,
    /// FAITHFUL paper §3.2.2 rule: a node gets d=0 ONLY if it lies on
    /// the GLOBAL mesh boundary. The chart's k-ring rim is NOT treated
    /// as a boundary (no kBoundaryRadiusFrac hack). Interior nodes —
    /// including those on this chart's truncation rim — get the full
    /// (α/4)h²I gap, exactly as in the paper's single global cloud.
    GlobalBoundaryOnly,
};

PolyReproResult run_chart_reproduction(
    const Eigen::MatrixXd&            V,
    const Eigen::MatrixXi&            F,
    int                               anchor,
    const std::vector<bool>&          is_global_boundary,
    Classifier                        cls   = Classifier::ShippedB1b,
    double                            alpha = 4.0,
    double                            beta  = 1.0,
    HMeasure                          hmode = HMeasure::Mean)
{
    const Eigen::Index n_v = V.rows();
    const auto adj         = diag_vertex_adjacency(static_cast<int>(n_v), F);
    const auto neighbours  = diag_k_ring(anchor, /*k_max=*/3, adj);
    // β_lme (locality) always uses the MEAN measure (unchanged from
    // production). Only the SME gap spacing h_gap is swept here.
    const auto h_a         = diag_vertex_h(V, adj);
    const auto h_gap       = (hmode == HMeasure::Max)
                               ? diag_vertex_h_max(V, adj)
                               : h_a;

    // β_lme = γ / h² with γ=1.6 (default).
    constexpr double gamma_lme = 1.6;
    Eigen::VectorXd beta_lme(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        beta_lme(a) = gamma_lme / (h_a(a) * h_a(a));
    }

    const Patch p = build_patch(anchor, V, neighbours, beta_lme);

    // Chart-outer-radius classification (matched the 2026-05-era
    // build_per_patch_sme_gaps; RETIRED there 2026-06-02 in favour of
    // the §3.2.2 global-boundary-frame graded recipe — kept here as
    // the era-history arm of the comparison).
    constexpr double kBoundaryRadiusFrac = 0.7;
    double xi_norm_max = 0.0;
    for (Eigen::Index k = 0; k < p.xi.rows(); ++k) {
        const double r = p.xi.row(k).norm();
        if (r > xi_norm_max) xi_norm_max = r;
    }
    const double r_thresh = kBoundaryRadiusFrac * xi_norm_max;

    std::vector<NodalGap> info;
    info.reserve(p.neighbor_ids.size());
    int n_bc = 0;
    for (Eigen::Index k = 0; k < p.xi.rows(); ++k) {
        const int gid = p.neighbor_ids[static_cast<std::size_t>(k)];
        NodalGap g{};
        g.h = h_gap(gid);
        bool is_corner = false;
        const bool is_global_bdry =
            is_global_boundary[static_cast<std::size_t>(gid)];
        if (cls == Classifier::ShippedB1b) {
            const bool is_chart_outer = p.xi.row(k).norm() >= r_thresh;
            is_corner = is_global_bdry || is_chart_outer;
        } else if (cls == Classifier::GlobalBoundaryOnly) {
            is_corner = is_global_bdry;
        }
        if (is_corner) {
            g.kind = NodalGapKind::BoundaryCorner;
            ++n_bc;
        } else {
            g.kind = NodalGapKind::Interior;
        }
        info.push_back(g);
    }
    const auto d_a = compute_nodal_gaps(info, alpha, beta);

    // Renormalise to h_chart=1, mirroring the assembler dispatch.
    const double h_chart = h_a(anchor);
    const double inv_h   = 1.0 / h_chart;
    const double inv_h2  = inv_h * inv_h;
    const Eigen::MatrixXd xi_s = p.xi * inv_h;
    std::vector<Eigen::Matrix2d> d_s = d_a;
    for (auto& dk : d_s) dk *= inv_h2;
    const double r_cut_s = 4.0 * 10.0;  // generous r_cut in scaled units

    // Polynomial sample values at chart nodes (in scaled coords).
    // M chosen so the "naive" reproduction error is meaningful: a
    // generic non-trace-free symmetric M of comparable norm to d_a.
    const Eigen::Matrix2d M_quad = (Eigen::Matrix2d() <<
        1.0, -0.25,
       -0.25,  1.5).finished();
    Eigen::VectorXd f_const(p.xi.rows());
    Eigen::VectorXd f_lin  (p.xi.rows());
    Eigen::VectorXd f_quad (p.xi.rows());
    for (Eigen::Index k = 0; k < p.xi.rows(); ++k) {
        const double x = xi_s(k, 0);
        const double y = xi_s(k, 1);
        f_const(k) = 1.0;
        f_lin  (k) = x + 2.0 * y - 3.0;
        const Eigen::Vector2d xv(x, y);
        f_quad (k) = xv.transpose() * M_quad * xv;
    }

    std::vector<Eigen::Vector2d> queries;
    queries.emplace_back(0.0, 0.0);
    const double r_q1 = 0.25 * xi_norm_max * inv_h;
    const double r_q2 = 0.50 * xi_norm_max * inv_h;
    for (double r : {r_q1, r_q2}) {
        queries.emplace_back( r,  0.0);
        queries.emplace_back(-r,  0.0);
        queries.emplace_back( 0.0,  r);
        queries.emplace_back( 0.0, -r);
    }

    PolyReproResult out{};
    out.anchor            = anchor;
    out.n_chart           = static_cast<int>(p.xi.rows());
    out.h_chart           = h_chart;
    out.xi_radius_max     = xi_norm_max;
    out.n_boundary_corner = n_bc;
    out.n_queries_tested  = static_cast<int>(queries.size());

    for (std::size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& xq = queries[qi];
        try {
            const auto bv = evaluate_sme_basis(
                xi_s, d_s, xq, r_cut_s,
                /*newton_tol=*/1e-10, /*max_iters=*/100);
            double f_hat_const = 0.0, f_hat_lin = 0.0, f_hat_quad = 0.0;
            // SME 2nd-moment residual: LHS - RHS where
            //   LHS = Σ s_a (x_a - x) ⊗ (x_a - x)
            //   RHS = Σ s_a d_a
            Eigen::Matrix2d lhs = Eigen::Matrix2d::Zero();
            Eigen::Matrix2d rhs = Eigen::Matrix2d::Zero();
            for (std::size_t k = 0; k < bv.indices.size(); ++k) {
                const int    a = bv.indices[k];
                const double s = bv.values[k];
                f_hat_const += s * f_const(a);
                f_hat_lin   += s * f_lin  (a);
                f_hat_quad  += s * f_quad (a);
                const Eigen::Vector2d u = xi_s.row(a).transpose() - xq;
                lhs += s * (u * u.transpose());
                rhs += s * d_s[static_cast<std::size_t>(a)];
            }
            const double x = xq(0), y = xq(1);
            const Eigen::Vector2d xv(x, y);
            const double f_const_true = 1.0;
            const double f_lin_true   = x + 2.0 * y - 3.0;
            const double f_quad_true  = xv.transpose() * M_quad * xv;
            out.max_err_const  = std::max(out.max_err_const,
                std::abs(f_hat_const - f_const_true));
            out.max_err_linear = std::max(out.max_err_linear,
                std::abs(f_hat_lin - f_lin_true));
            out.max_err_naive_quad = std::max(out.max_err_naive_quad,
                std::abs(f_hat_quad - f_quad_true));
            out.max_residual_2nd_mom = std::max(out.max_residual_2nd_mom,
                (lhs - rhs).lpNorm<Eigen::Infinity>());
            out.max_d_slack_trace = std::max(out.max_d_slack_trace,
                std::abs(rhs.trace()));
        } catch (const std::exception&) {
            ++out.n_queries_diverged;
            if (qi == 0)      ++out.n_div_center;
            else if (qi <= 4) ++out.n_div_inner;
            else              ++out.n_div_outer;
        }
    }
    return out;
}

void dump_result(const char* mesh_name, const char* tag,
                  const PolyReproResult& r)
{
    std::fprintf(stderr,
        "[sme_chart_diag][%s/%s] anchor=%d n=%d h=%.3g "
        "n_BC=%d/%d  diverged=%d/%d\n"
        "  PoU=%.2e  1st-mom=%.2e  SME-2nd-mom-residual=%.2e  "
        "naive-quad-err=%.2e (~tr(M·d̄)=%.2e)\n",
        mesh_name, tag, r.anchor, r.n_chart, r.h_chart,
        r.n_boundary_corner, r.n_chart,
        r.n_queries_diverged, r.n_queries_tested,
        r.max_err_const, r.max_err_linear,
        r.max_residual_2nd_mom,
        r.max_err_naive_quad, r.max_d_slack_trace);
}

}  // namespace

TEST_CASE("SME diag: per-call timing breakdown on a polar disk chart",
          "[.diag][lme][sme][diag_sme_timing]")
{
    // Per-call cost breakdown for the SME basis on a representative
    // chart from the 32x4 polar disk fixture (where the timing test
    // measured SME K=520 ms vs LME-curved K=57 ms — 9.2x). Compare:
    //   evaluate_basis                       (LME, values only)
    //   evaluate_basis_grad_and_hess         (LME, with Hessian)
    //   evaluate_sme_basis                   (SME, values only)
    //   evaluate_sme_basis_and_grad          (SME, with gradient)
    //   evaluate_sme_basis_grad_and_hess     (SME, with Hessian)
    //
    // Ratios tell us where the K-assembly time goes:
    //   SME-values / LME-values:                M-path slowdown
    //   SME-hess / SME-gradient:                FD-Hessian cost
    //   SME-hess / SME-values:                  Combined slowdown
    //   SME-gradient / SME-values:              IFT overhead
    // Synthetic 5x5 unit-spacing grid on [-2, 2]^2 — the same
    // fixture the SME basis unit tests use. Well-conditioned for
    // both LME and SME; isolates per-call costs without polar-disk
    // chart degeneracies.
    constexpr int kN = 5;
    constexpr double kH = 1.0;
    Eigen::MatrixXd xi_s(kN * kN, 2);
    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            xi_s(j * kN + i, 0) = -2.0 + static_cast<double>(i) * kH;
            xi_s(j * kN + i, 1) = -2.0 + static_cast<double>(j) * kH;
        }
    }
    Eigen::VectorXd beta_chart(kN * kN);
    constexpr double gamma_lme = 1.6;
    beta_chart.setConstant(gamma_lme / (kH * kH));

    // Build d_a using the paper §3.2.2 recipe (same as the SME
    // basis unit test). 4 corners → BoundaryCorner, edges →
    // BoundaryEdgeMid / Near, interior → Interior.
    std::vector<NodalGap> info(kN * kN);
    for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kN; ++i) {
            const int idx = j * kN + i;
            info[idx].h = kH;
            const bool ox = (i == 0 || i == kN - 1);
            const bool oy = (j == 0 || j == kN - 1);
            if (ox && oy) {
                info[idx].kind = NodalGapKind::BoundaryCorner;
            } else if (ox) {
                info[idx].kind = NodalGapKind::BoundaryEdgeMid;
                info[idx].t = Eigen::Vector2d(0.0, 1.0);
            } else if (oy) {
                info[idx].kind = NodalGapKind::BoundaryEdgeMid;
                info[idx].t = Eigen::Vector2d(1.0, 0.0);
            } else {
                info[idx].kind = NodalGapKind::Interior;
            }
        }
    }
    auto d_a = compute_nodal_gaps(info, 4.0, 1.0);
    // No renormalisation here — kH=1.0 already.

    // Per-call timings: many repetitions to drown out one-off noise.
    constexpr int    n_reps    = 2000;
    constexpr double r_cut_lme = 10.0;
    constexpr double r_cut_sme = 10.0;

    // Query points: 10x10 grid strictly between nodes (LME's
    // grad/Hess throw at node coincidence). Off-set by 0.5*kH/9
    // from each node row.
    std::vector<Eigen::Vector2d> qs;
    qs.reserve(100);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            const double xq = -0.95 + 1.9 * (i / 9.0);
            const double yq = -0.95 + 1.9 * (j / 9.0);
            qs.emplace_back(xq, yq);
        }
    }
    const auto& qs_lme = qs;
    const auto& qs_sme = qs;

    auto t_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto t_us  = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    auto bench = [&](const char* name,
                      const std::vector<Eigen::Vector2d>& qs,
                      auto func) {
        const auto t0 = t_now();
        std::size_t n_calls = 0;
        for (int rep = 0; rep < n_reps; ++rep) {
            for (const auto& xq : qs) {
                func(xq);
                ++n_calls;
            }
        }
        const auto t1 = t_now();
        const double dt = t_us(t0, t1);
        std::fprintf(stderr,
            "[sme_timing] %-38s  total=%9.1f ms  per-call=%7.2f us  "
            "(%zu calls)\n",
            name, dt / 1000.0, dt / static_cast<double>(n_calls),
            n_calls);
    };

    constexpr double newton_tol = 1e-10;
    constexpr int    newton_max = 80;

    bench("LME evaluate_basis",               qs_lme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_basis(xi_s, beta_chart, xq, r_cut_lme,
                               newton_tol, newton_max);
    });
    bench("LME evaluate_basis_grad_and_hess", qs_lme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_basis_grad_and_hess(xi_s, beta_chart, xq, r_cut_lme,
                                              newton_tol, newton_max);
    });
    bench("SME evaluate_sme_basis",           qs_sme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_sme_basis(xi_s, d_a, xq, r_cut_sme,
                                    newton_tol, newton_max);
    });
    bench("SME evaluate_sme_basis_and_grad",  qs_sme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_sme_basis_and_grad(xi_s, d_a, xq, r_cut_sme,
                                              newton_tol, newton_max);
    });
    bench("SME evaluate_sme_basis_grad_and_hess", qs_sme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_sme_basis_grad_and_hess(xi_s, d_a, xq, r_cut_sme,
                                                   newton_tol, newton_max);
    });
    bench("SME ..._grad_and_hess_closed_form", qs_sme,
          [&](const Eigen::Vector2d& xq) {
        return evaluate_sme_basis_grad_and_hess_closed_form(
            xi_s, d_a, xq, r_cut_sme, newton_tol, newton_max);
    });

    SUCCEED("timing breakdown on stderr — see [sme_timing] lines");
}

TEST_CASE("SME diag: polynomial reproduction on icosphere k=2 chart vs "
          "polar disk 32x8 chart",
          "[.diag][lme][sme][diag_sme_chart]")
{
    // --- icosphere k=2: pick interior anchors (no global boundary). ---
    {
        const auto mesh = chladni::mesh::generate_icosphere(/*r=*/0.10,
                                                            /*sub=*/2);
        const Eigen::Index n_v = mesh.V.rows();
        const std::vector<bool> no_bdry(static_cast<std::size_t>(n_v),
                                         false);
        for (int anchor : {0, 50, 100, 161}) {
            if (anchor >= static_cast<int>(n_v)) continue;
            for (auto cls : {Classifier::ShippedB1b,
                              Classifier::AllInterior}) {
                const auto r = run_chart_reproduction(
                    mesh.V, mesh.F, anchor, no_bdry, cls);
                dump_result("icosphere_k2",
                    cls == Classifier::ShippedB1b ? "shipped" : "all_int",
                    r);
            }
        }
    }

    // --- polar disk 32x8: pick interior + near-boundary anchors. ---
    {
        const auto mesh = chladni::mesh::generate_circular_disk(
            /*R=*/0.10, /*n_az=*/32, /*n_rad=*/8);
        const Eigen::Index n_v = mesh.V.rows();
        const auto edges = chladni::shell::build_edges(mesh.F);
        std::vector<bool> is_bdry(static_cast<std::size_t>(n_v), false);
        for (const auto& e : edges) {
            if (e.is_boundary()) {
                is_bdry[static_cast<std::size_t>(e.v0)] = true;
                is_bdry[static_cast<std::size_t>(e.v1)] = true;
            }
        }
        for (int anchor : {0, 64, 128, 200}) {
            if (anchor >= static_cast<int>(n_v)) continue;
            for (auto cls : {Classifier::ShippedB1b,
                              Classifier::AllInterior}) {
                const auto r = run_chart_reproduction(
                    mesh.V, mesh.F, anchor, is_bdry, cls);
                dump_result("polar_disk_32x8",
                    cls == Classifier::ShippedB1b ? "shipped" : "all_int",
                    r);
            }
        }
    }

    // The test itself is informational — no REQUIRE. The diagnostic
    // signal is in the dump_result lines on stderr.
    SUCCEED("diagnostic dump on stderr — see [sme_chart_diag] lines");
}

// =====================================================================
// AUDIT 2026-06-01: does the gap spacing measure (MEAN vs MAX incident
// edge) drive the α=2 in-chart divergence? Paper §3.2.1 uses the MAX
// adjacent spacing for the interior gap; production uses the MEAN
// (vertex_one_ring_h). Sweep (h-measure × α) over every interior anchor
// of two production meshes and total the in-chart Newton divergences.
// =====================================================================
TEST_CASE("SME diag: divergence vs gap-spacing measure (mean/max) and alpha",
          "[.diag][lme][sme][diag_sme_hmeasure]")
{
    struct Mesh { const char* name; chladni::mesh::TriMesh m; };
    std::vector<Mesh> meshes;
    meshes.push_back({"polar_disk_32x8",
        chladni::mesh::generate_circular_disk(0.10, 32, 8)});
    meshes.push_back({"icosphere_k3",
        chladni::mesh::generate_icosphere(0.10, 3)});

    for (auto& mesh : meshes) {
        const Eigen::Index n_v = mesh.m.V.rows();
        const auto edges = chladni::shell::build_edges(mesh.m.F);
        std::vector<bool> is_bdry(static_cast<std::size_t>(n_v), false);
        for (const auto& e : edges) {
            if (e.is_boundary()) {
                is_bdry[static_cast<std::size_t>(e.v0)] = true;
                is_bdry[static_cast<std::size_t>(e.v1)] = true;
            }
        }

        struct ClsCase { const char* name; Classifier cls; };
        for (ClsCase cc : {ClsCase{"shipped-hack", Classifier::ShippedB1b},
                           ClsCase{"faithful-global",
                                   Classifier::GlobalBoundaryOnly}}) {
            for (double alpha : {2.0, 4.0}) {
                int dc = 0, di = 0, do_ = 0, total_q = 0, n_anchors = 0;
                for (int a = 0; a < static_cast<int>(n_v); ++a) {
                    // Skip global-boundary anchors (no interior chart).
                    if (is_bdry[static_cast<std::size_t>(a)]) continue;
                    const auto r = run_chart_reproduction(
                        mesh.m.V, mesh.m.F, a, is_bdry,
                        cc.cls, alpha, 1.0, HMeasure::Mean);
                    dc += r.n_div_center;
                    di += r.n_div_inner;
                    do_ += r.n_div_outer;
                    total_q += r.n_queries_tested;
                    ++n_anchors;
                }
                std::fprintf(stderr,
                    "[sme_radclass][%s] %-16s alpha=%.1f : "
                    "diverge center=%d inner(0.25)=%d outer(0.50)=%d "
                    "(of %d anchors, %d q)\n",
                    mesh.name, cc.name, alpha,
                    dc, di, do_, n_anchors, total_q);
            }
        }
    }
    SUCCEED("see [sme_hmeasure] lines on stderr");
}
