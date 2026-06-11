/**
 * @file lme.cpp
 * @brief Implementation of the 1st-order Local Max-Ent basis evaluator.
 *
 * Solves the convex dual Newton minimisation for @f$ \lambda^\star(x) @f$
 * and produces the basis values
 * @f$ p_a(x) = q_a / Z(x, \lambda^\star) @f$ following
 * @cite arroyo_ortiz_2006_local_maximum_entropy (eqs. 7 & 10).
 *
 * Spatial gradients use the same paper's eq. 44 closed form
 *
 *   @f[
 *     \nabla p_a^\star(x) = -p_a^\star\, (J^\star)^{-1}\,(x - x_a)
 *   @f]
 *
 * (uniform-@f$\beta@f$ specialisation; the @f$\nabla\beta@f$ term in
 * Arroyo--Ortiz eq. 44 drops out for spatially-constant @f$\beta@f$).
 * Both @ref chladni::shell::lme::evaluate_basis and
 * @ref chladni::shell::lme::evaluate_basis_and_grad share the same dual
 * Newton solve via a private @c LMEState helper.
 *
 * The @ref chladni::shell::LMEAssembler::assemble_M (Galerkin mass) and
 * @ref chladni::shell::LMEAssembler::assemble_K (flat-plate Kirchhoff
 * bending @em plus 2D plane-stress membrane on the in-plane components)
 * are both shipped. Curved-shell support (Millan 2011 wPCA + Shepard
 * PoU) is also shipped and is the default assembly path
 * (@ref chladni::shell::LMEAssembler::Params::use_curved_shell).
 *
 * @section dual_newton Dual Newton (Arroyo--Ortiz eq. 10)
 *
 * For each query @f$ x \in \mathrm{conv}\,X @f$ we minimise
 *
 *   @f[
 *     \mathcal F(\lambda) = \ln Z(x, \lambda)
 *     = \ln \sum_a \exp\!\bigl(
 *         -\beta_a |x - x_a|^2 + \lambda \cdot (x - x_a)
 *       \bigr).
 *   @f]
 *
 * Gradient and Hessian in @f$ \mathbb R^d @f$ are
 *
 *   @f{aligned}{
 *     \nabla \mathcal F &= \sum_a p_a(x, \lambda)\, m_a, \\
 *     \nabla^2 \mathcal F &=
 *       \sum_a p_a(x, \lambda)\, m_a m_a^\top
 *       - \bigl(\sum_a p_a m_a\bigr)\!\bigl(\sum_a p_a m_a\bigr)^\top
 *   @f}
 *
 * with @f$ m_a = x - x_a @f$. The Hessian is the Fisher information
 * matrix of the discrete distribution @f$ \{p_a\} @f$ — positive
 * semi-definite, positive definite when the active nodes' offsets
 * @f$ \{m_a\} @f$ span @f$ \mathbb R^d @f$. Newton's method converges
 * quadratically from @f$ \lambda_0 = 0 @f$.
 *
 * @section stability Numerical stability
 *
 * The log-weights @f$ -\beta_a |m_a|^2 + \lambda \cdot m_a @f$ can drift
 * by tens-to-hundreds in magnitude across the active set, so the
 * implementation uses the standard log-sum-exp shift: subtract
 * @f$ \max_a f_a @f$ before exponentiation. The shift cancels in the
 * normalised @f$ p_a = q_a / \sum q_a @f$.
 *
 * Conv-hull corners are a special case: the Hessian degenerates because
 * the active @f$ \{m_a\} @f$ collapse onto a half-plane (all nonneg
 * offsets in one direction). To dodge the rank deficiency we check up
 * front whether @p x coincides with any node and short-circuit to the
 * delta distribution @f$ p_b = 1 @f$, recovering the weak
 * Kronecker-delta property of the basis (Arroyo--Ortiz §3.1) exactly.
 * The gradient evaluator @ref chladni::shell::lme::evaluate_basis_and_grad
 * throws @c std::domain_error in this case (the gradient is not
 * well-defined at conv-hull corners).
 */

#include <chladni/shell/lme.hpp>

#include <Eigen/Dense>  // matrix inverse for the dxd Hessian solve

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace chladni::shell {

// Forward declaration — used by LMEAssembler::assemble_M below.
[[nodiscard]] std::vector<QuadraturePoint> quadrature_points(
    QuadratureRule rule);

namespace {

/// Run @p worker over @p n_threads shards (n_threads-1 spawned threads
/// plus the calling thread running the last shard, matching the
/// assemblers' "don't waste a worker" pattern) and **propagate worker
/// exceptions safely**.
///
/// An exception thrown inside a `std::thread` callable that is not
/// caught within that callable invokes `std::terminate` — a hard
/// `SIGABRT` the caller cannot catch. The per-Gauss-point assemblers
/// legitimately throw (e.g. "every Shepard-active patch's in-chart
/// Newton diverged" on an infeasible r_cut / gamma / mesh combination),
/// and those throws must reach the caller as a normal
/// `std::exception` so the GUI can fall back to the legacy path instead
/// of the process dying. This helper catches the first worker
/// exception, joins all threads, then rethrows it on the calling
/// thread.
void run_threaded(int n_threads, const std::function<void(int)>& worker)
{
    std::exception_ptr first_exc;
    std::mutex         exc_mutex;
    auto guarded = [&](int tid) {
        try {
            worker(tid);
        } catch (...) {
            const std::lock_guard<std::mutex> lk(exc_mutex);
            if (!first_exc) {
                first_exc = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(std::max(0, n_threads - 1)));
    for (int tid = 0; tid < n_threads - 1; ++tid) {
        threads.emplace_back(guarded, tid);
    }
    guarded(n_threads - 1);
    for (auto& th : threads) {
        th.join();
    }
    if (first_exc) {
        std::rethrow_exception(first_exc);
    }
}

}  // namespace

namespace lme {

Eigen::RowVector3d reflect_across_edge_line(
    const Eigen::RowVector3d& v0,
    const Eigen::RowVector3d& v1,
    const Eigen::RowVector3d& v_int)
{
    const Eigen::RowVector3d e   = v1 - v0;
    const double             e2  = e.squaredNorm();
    if (!(e2 > 0.0)) {
        throw std::invalid_argument(
            "lme::reflect_across_edge_line: v0 and v1 coincide; "
            "edge has zero length, cannot reflect.");
    }
    const double t   = (v_int - v0).dot(e) / e2;
    const Eigen::RowVector3d proj = v0 + t * e;
    return 2.0 * proj - v_int;
}

Eigen::MatrixXd build_ghost_positions(
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::MatrixXi&                   F,
    const std::vector<BoundaryEdge>&         bdry)
{
    const Eigen::Index G = static_cast<Eigen::Index>(bdry.size());
    Eigen::MatrixXd ghosts(G, 3);
    if (G == 0) {
        return ghosts;
    }

    // Boundary-vertex flags + one-ring neighbour lists restricted to
    // the boundary vertices (the only ones the δ probe reads). One
    // O(F) pass; duplicates are harmless under the max below.
    std::vector<bool> is_bdry(static_cast<std::size_t>(V.rows()), false);
    for (const auto& b : bdry) {
        is_bdry[static_cast<std::size_t>(b.v0)] = true;
        is_bdry[static_cast<std::size_t>(b.v1)] = true;
    }
    std::vector<std::vector<int>> bdry_nbrs(
        static_cast<std::size_t>(V.rows()));
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int i = F(t, k);
            const int j = F(t, (k + 1) % 3);
            if (is_bdry[static_cast<std::size_t>(i)]) {
                bdry_nbrs[static_cast<std::size_t>(i)].push_back(j);
            }
            if (is_bdry[static_cast<std::size_t>(j)]) {
                bdry_nbrs[static_cast<std::size_t>(j)].push_back(i);
            }
        }
    }

    for (Eigen::Index i = 0; i < G; ++i) {
        const auto& b = bdry[static_cast<std::size_t>(i)];
        const Eigen::RowVector3d x0 = V.row(b.v0);
        const Eigen::RowVector3d x1 = V.row(b.v1);
        const Eigen::RowVector3d e  = x1 - x0;
        const double             e2 = e.squaredNorm();
        if (!(e2 > 0.0)) {
            throw std::invalid_argument(
                "lme::build_ghost_positions: boundary edge (" +
                std::to_string(b.v0) + ", " + std::to_string(b.v1) +
                ") has zero length.");
        }
        const Eigen::RowVector3d m = 0.5 * (x0 + x1);

        // collect_boundary_edges leaves v_int = -1 when the adjacent
        // triangle has no third corner distinct from (v0, v1) (a degenerate
        // face); V.row(-1) is UB. Unreachable on manifold input, but guard
        // it explicitly like the other degeneracy checks here, mirroring the
        // `if (be.v_int >= 0)` guard in compute_sme_boundary_frames.
        if (b.v_int < 0) {
            throw std::invalid_argument(
                "lme::build_ghost_positions: boundary edge (" +
                std::to_string(b.v0) + ", " + std::to_string(b.v1) +
                ") has no interior vertex (degenerate adjacent face).");
        }

        // In-plane INWARD direction: the component of (v_int - m)
        // perpendicular to the edge. Its negation is the outward
        // normal the ghost is offset along; its norm is the old
        // reflection offset (the δ fallback below).
        const Eigen::RowVector3d w_int = V.row(b.v_int) - m;
        const Eigen::RowVector3d n_in  = w_int - (w_int.dot(e) / e2) * e;
        const double             d_int = n_in.norm();
        if (!(d_int > 0.0)) {
            throw std::invalid_argument(
                "lme::build_ghost_positions: boundary edge (" +
                std::to_string(b.v0) + ", " + std::to_string(b.v1) +
                ") has a degenerate (collinear) adjacent triangle.");
        }
        const Eigen::RowVector3d n_out = -n_in / d_int;

        // First-interior-row directional spacing: the RMA13 §3.2.1
        // max-adjacent rule h_a(u) = max_w |(x_w - x_a)·u| evaluated at
        // the two boundary anchors a ∈ {v0, v1} along the boundary
        // normal u = n_out, restricted to interior neighbours w. The
        // difference is anchored at each neighbour's OWN boundary
        // vertex (not the edge midpoint): on a curved rim a diagonal
        // neighbour's midpoint-relative normal component picks up the
        // chord sagitta of its azimuthal offset and overshoots the row
        // spacing by O(25%). Falls back to the v_int reflection offset
        // when no interior neighbour exists (degenerate strip meshes).
        double delta = 0.0;
        for (const int v : {b.v0, b.v1}) {
            for (const int w : bdry_nbrs[static_cast<std::size_t>(v)]) {
                if (is_bdry[static_cast<std::size_t>(w)]) continue;
                const double proj =
                    std::abs((V.row(w) - V.row(v)).dot(n_out));
                delta = std::max(delta, proj);
            }
        }
        if (!(delta > 0.0)) {
            delta = d_int;
        }

        ghosts.row(i) = m + delta * n_out;
    }
    return ghosts;
}

std::vector<BoundaryEdge> collect_boundary_edges(const Eigen::MatrixXi& F)
{
    const auto edges = chladni::shell::build_edges(F);
    std::vector<BoundaryEdge> result;
    result.reserve(edges.size());

    for (const auto& e : edges) {
        if (!e.is_boundary()) continue;

        // The non-(-1) face is the adjacent triangle. Edge stores
        // face_left for boundary edges by convention (build_edges
        // guarantees) but we accept either to be defensive against
        // future winding changes.
        const Eigen::Index f =
            (e.face_left != -1) ? e.face_left : e.face_right;

        // The third corner of the adjacent triangle — the only
        // F(f, k) that is neither v0 nor v1.
        int v_int = -1;
        for (int k = 0; k < 3; ++k) {
            const int v = F(static_cast<Eigen::Index>(f), k);
            if (v != static_cast<int>(e.v0) && v != static_cast<int>(e.v1)) {
                v_int = v;
                break;
            }
        }

        result.push_back(BoundaryEdge{
            static_cast<int>(e.v0),
            static_cast<int>(e.v1),
            v_int,
            static_cast<int>(f),
        });
    }
    return result;
}

std::vector<Eigen::Matrix2d> compute_nodal_gaps(
    const std::vector<NodalGap>& info,
    double                       alpha,
    double                       beta)
{
    if (!(alpha > 1.0)) {
        throw std::invalid_argument(
            "lme::compute_nodal_gaps: alpha must be > 1 (got " +
            std::to_string(alpha) + ").");
    }
    if (!(beta >= 1.0)) {
        throw std::invalid_argument(
            "lme::compute_nodal_gaps: beta must be >= 1 (got " +
            std::to_string(beta) + ").");
    }

    // Tolerance for unit-length and orthogonality checks on
    // tangent/normal/eigenvector inputs. Anything tighter trips on
    // floating-point representations of common irrational unit vectors
    // (e.g. cos/sin of arbitrary angles); anything looser stops
    // catching user typos like "(2, 0)".
    constexpr double kUnitTol  = 1e-9;
    const double     a_quarter = alpha / 4.0;

    auto require_positive =
        [](double v, const char* name, std::size_t k) {
            if (!(v > 0.0)) {
                throw std::invalid_argument(
                    "lme::compute_nodal_gaps: NodalGap[" +
                    std::to_string(k) + "]." + name +
                    " must be > 0 (got " + std::to_string(v) + ").");
            }
        };
    auto require_unit =
        [](const Eigen::Vector2d& v, const char* name, std::size_t k) {
            const double n = v.norm();
            if (std::abs(n - 1.0) > kUnitTol) {
                throw std::invalid_argument(
                    "lme::compute_nodal_gaps: NodalGap[" +
                    std::to_string(k) + "]." + name +
                    " must be unit-length (||" + name + "|| = " +
                    std::to_string(n) + ").");
            }
        };

    std::vector<Eigen::Matrix2d> out;
    out.reserve(info.size());

    for (std::size_t k = 0; k < info.size(); ++k) {
        const NodalGap& g = info[k];
        Eigen::Matrix2d d = Eigen::Matrix2d::Zero();

        switch (g.kind) {
        case NodalGapKind::Interior: {
            require_positive(g.h, "h", k);
            d = a_quarter * g.h * g.h * Eigen::Matrix2d::Identity();
            break;
        }
        case NodalGapKind::InteriorAnisotropic: {
            require_positive(g.h1, "h1", k);
            require_positive(g.h2, "h2", k);
            require_unit(g.v1, "v1", k);
            require_unit(g.v2, "v2", k);
            const double v1v2 = std::abs(g.v1.dot(g.v2));
            if (v1v2 > kUnitTol) {
                throw std::invalid_argument(
                    "lme::compute_nodal_gaps: NodalGap[" +
                    std::to_string(k) +
                    "].v1 and v2 must be orthogonal (|v1.v2| = " +
                    std::to_string(v1v2) + ").");
            }
            d = a_quarter *
                ( g.h1 * g.h1 * (g.v1 * g.v1.transpose())
                + g.h2 * g.h2 * (g.v2 * g.v2.transpose()) );
            break;
        }
        case NodalGapKind::BoundaryCorner: {
            // Paper §3.2.2: corners interpolate exactly; the slack
            // matrix is the zero matrix, recovering the same property
            // 1st-order LME already enjoys at conv-hull corners.
            break;
        }
        case NodalGapKind::BoundaryEdgeMid: {
            require_positive(g.h, "h", k);
            require_unit(g.t, "t", k);
            // Directional tangential spacing (RMA13 §3.2.2); falls back to
            // the isotropic h on quasi-uniform boundaries (h_t == 0).
            const double ht = (g.h_t > 0.0) ? g.h_t : g.h;
            d = a_quarter * ht * ht * (g.t * g.t.transpose());
            break;
        }
        case NodalGapKind::BoundaryEdgeNearCorner: {
            require_positive(g.h, "h", k);
            require_unit(g.t, "t", k);
            const double ht = (g.h_t > 0.0) ? g.h_t : g.h;
            d = beta * ht * ht * (g.t * g.t.transpose());
            break;
        }
        case NodalGapKind::NearOneBoundaryEdge: {
            require_positive(g.h, "h", k);
            require_unit(g.t, "t", k);
            require_unit(g.n, "n", k);
            // β h_n² along the (possibly sparse) normal direction + (α/4) h_t²
            // along the tangent — each with its own directional spacing.
            const double hn = (g.h_n > 0.0) ? g.h_n : g.h;
            const double ht = (g.h_t > 0.0) ? g.h_t : g.h;
            d =   beta      * hn * hn * (g.n * g.n.transpose())
                + a_quarter * ht * ht * (g.t * g.t.transpose());
            break;
        }
        case NodalGapKind::NearTwoBoundaryEdges: {
            require_positive(g.h, "h", k);
            require_unit(g.n,  "n",  k);
            require_unit(g.n2, "n2", k);
            const double hn = (g.h_n > 0.0) ? g.h_n : g.h;
            d = beta * hn * hn *
                (g.n * g.n.transpose() + g.n2 * g.n2.transpose());
            break;
        }
        }

        out.push_back(d);
    }
    return out;
}

namespace {

/// Tolerance for "d_b is the zero matrix" exact-match: at corners the
/// SME program degenerates to s = δ_b (paper §3.1 — moment-space
/// vertex coincides with x); Newton would diverge so we short-circuit.
constexpr double kSmeZeroDTol = 1e-14;
/// "x coincides with node x_b" — mirrors the LME constant declared
/// further down. Squared-distance threshold.
constexpr double kSmeNodeEps2 = 1.0e-28;

/// State produced by the converged SME Newton solve. Shared between
/// @ref evaluate_sme_basis and its gradient/Hessian counterparts.
/// On the corner exact-match shortcut @c exact_match is @c true and
/// only @c exact_match_id is populated.
struct SMEState {
    bool             exact_match     = false;
    int              exact_match_id  = -1;
    std::vector<int> active;       ///< Global node ids in active set.
    Eigen::MatrixXd  PHI;          ///< n_act x 5 — per-node φ_a(x).
    Eigen::VectorXd  p;            ///< n_act — basis values s_a(x).
    Eigen::VectorXd  theta;        ///< 5 — converged dual (λ, μ_11, μ_22, μ_12).
    Eigen::MatrixXd  H;            ///< 5x5 — unridged converged Hessian
                                   ///<     (Fisher info; used by IFT).
};

/// Run input validation + active-set selection + Newton on the SME
/// dual. Returns enough state to compute basis values, gradients, and
/// Hessians downstream. @p caller goes into error messages so the
/// public function can be identified.
SMEState compute_sme_state(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double      r_cut,
    double      newton_tol,
    int         newton_max_iters,
    const char* caller,
    const Eigen::VectorXd* theta_warm_start = nullptr)
{
    if (nodes.rows() == 0) {
        throw std::invalid_argument(
            std::string("lme::") + caller + ": nodes matrix is empty");
    }
    if (nodes.cols() != 2) {
        throw std::invalid_argument(
            std::string("lme::") + caller + ": this entry point "
            "supports only 2D charts (nodes.cols() must equal 2; got " +
            std::to_string(nodes.cols()) + ")");
    }
    if (static_cast<Eigen::Index>(d.size()) != nodes.rows()) {
        throw std::invalid_argument(
            std::string("lme::") + caller +
            ": d.size() must equal nodes.rows() (got d.size() = " +
            std::to_string(d.size()) + ", nodes.rows() = " +
            std::to_string(nodes.rows()) + ")");
    }
    if (x.size() != 2) {
        throw std::invalid_argument(
            std::string("lme::") + caller +
            ": x must have dimension 2 (got " +
            std::to_string(x.size()) + ")");
    }
    if (!(r_cut > 0.0)) {
        throw std::invalid_argument(
            std::string("lme::") + caller +
            ": r_cut must be strictly positive");
    }
    // d_a symmetry — surface a user-friendly error early; PSD check
    // stays implicit (Newton diverges on indefinite d).
    constexpr double kSymTol = 1e-12;
    for (std::size_t a = 0; a < d.size(); ++a) {
        const Eigen::Matrix2d& da = d[a];
        if (std::abs(da(0, 1) - da(1, 0)) > kSymTol) {
            throw std::invalid_argument(
                std::string("lme::") + caller + ": d[" +
                std::to_string(a) +
                "] is not symmetric (|d_12 - d_21| = " +
                std::to_string(std::abs(da(0, 1) - da(1, 0))) + ")");
        }
    }

    const Eigen::Index n_nodes = nodes.rows();
    const double       r_cut2  = r_cut * r_cut;

    SMEState s;
    s.active.reserve(static_cast<std::size_t>(n_nodes));
    int    closest_node  = -1;
    double closest_dist2 = std::numeric_limits<double>::infinity();
    for (Eigen::Index a = 0; a < n_nodes; ++a) {
        const double dist2 =
            (x.transpose() - nodes.row(a)).squaredNorm();
        if (dist2 <= kSmeNodeEps2 &&
            d[static_cast<std::size_t>(a)].cwiseAbs().maxCoeff() <
                kSmeZeroDTol)
        {
            s.exact_match    = true;
            s.exact_match_id = static_cast<int>(a);
            return s;
        }
        if (dist2 <= r_cut2) {
            s.active.push_back(static_cast<int>(a));
            if (dist2 < closest_dist2) {
                closest_dist2 = dist2;
                closest_node  = static_cast<int>(a);
            }
        }
    }
    if (s.active.empty()) {
        throw std::invalid_argument(
            std::string("lme::") + caller +
            ": no nodes within r_cut of x — "
            "increase r_cut or check that x lies inside the node "
            "cloud");
    }

    const Eigen::Index n_act =
        static_cast<Eigen::Index>(s.active.size());

    // Per-node 5-vector. θ = (λ_1, λ_2, μ_11, μ_22, μ_12); pack each
    // active node's contribution log w_a = θ · φ_a into row k of PHI:
    //   φ_a = (u_x, u_y, -D_a,11, -D_a,22, -2 D_a,12),
    // with u_a = x - x_a and D_a = u_a u_a^T - d_a. The factor of 2
    // on the off-diagonal absorbs μ : D_a = μ_11 D_11 + μ_22 D_22 +
    // 2 μ_12 D_12 (μ_12 = μ_21).
    s.PHI.resize(n_act, 5);
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const int a = s.active[static_cast<std::size_t>(k)];
        const Eigen::Vector2d u =
            x - nodes.row(a).transpose();
        const Eigen::Matrix2d D =
            u * u.transpose() - d[static_cast<std::size_t>(a)];
        s.PHI(k, 0) =  u(0);
        s.PHI(k, 1) =  u(1);
        s.PHI(k, 2) = -D(0, 0);
        s.PHI(k, 3) = -D(1, 1);
        s.PHI(k, 4) = -2.0 * D(0, 1);
    }

    // Newton on θ ∈ R^5 with three stabilisations (all needed for
    // queries < ~0.6 h from the chart boundary; see Phase A.2 commit
    // message for the empirical justification):
    //   (i)   Scale-relative Tikhonov ridge   max(1e-14, 1e-8·tr H).
    //   (ii)  Step-norm cap                   |dθ| ≤ 10.
    //   (iii) Armijo backtracking on ln Z.
    constexpr double kArmijoC1      = 1.0e-4;
    constexpr double kMinStep       = 1.0e-12;
    constexpr double kMaxDthetaNorm = 10.0;

    auto log_sum_exp_minus_logZ = [&](const Eigen::VectorXd& theta_arg,
                                      Eigen::VectorXd&       p_out)
        -> double
    {
        Eigen::VectorXd f = s.PHI * theta_arg;
        const double f_max = f.maxCoeff();
        f.array() -= f_max;
        p_out = f.array().exp();
        const double sum = p_out.sum();
        p_out /= sum;
        return f_max + std::log(sum);
    };

    // Warm-start support: callers that already have a good θ guess
    // (e.g. evaluate_sme_basis_grad_and_hess's at-x converged θ
    // re-used for the 4 FD-perturbed queries at x ± h e_l, h=1e-5
    // — see the FD layer below) pass it here so Newton converges
    // in 1-2 iters instead of 8-15 cold-start.
    //
    // Otherwise use the PAPER's initial guess (RMA13 §3.4.3, closing
    // audit gap A6): λ_est = 0, μ_est = ½(d_CN)⁻¹ with d_CN the gap
    // matrix of the node CLOSEST to the evaluation point, via the
    // pseudoinverse where d_CN is singular ("on or next to the
    // boundaries ... det(d_CN) = 0", in which case the singular
    // direction's μ component is left at 0, matching the paper's 1D
    // boundary recipe μ_est = 0). The paper measures 30–40 % fewer
    // Newton iterations vs the null guess. NOTE this is a per-point
    // ANALYTIC estimate — unrelated to the neighbour-to-neighbour θ
    // reuse across Gauss points, which was tried and measured NEGATIVE
    // for the LME path and never wired (no warm-start knob exists).
    if (theta_warm_start != nullptr && theta_warm_start->size() == 5) {
        s.theta = *theta_warm_start;
    } else {
        s.theta = Eigen::VectorXd::Zero(5);
        if (closest_node >= 0) {
            const Eigen::Matrix2d& d_cn =
                d[static_cast<std::size_t>(closest_node)];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(d_cn);
            if (es.info() == Eigen::Success) {
                const Eigen::Vector2d ev = es.eigenvalues();
                const double tol_ev =
                    1.0e-12 * std::max(1.0, ev.cwiseAbs().maxCoeff());
                Eigen::Vector2d inv_ev = Eigen::Vector2d::Zero();
                for (int i = 0; i < 2; ++i) {
                    if (ev(i) > tol_ev) inv_ev(i) = 1.0 / ev(i);
                }
                const Eigen::Matrix2d mu_est =
                    0.5 * es.eigenvectors() * inv_ev.asDiagonal()
                        * es.eigenvectors().transpose();
                s.theta(2) = mu_est(0, 0);
                s.theta(3) = mu_est(1, 1);
                s.theta(4) = mu_est(0, 1);
            }
        }
    }
    s.p.resize(n_act);
    double f_curr = log_sum_exp_minus_logZ(s.theta, s.p);

    // RELATIVE convergence scale. The dual gradient g = Σ p_a φ_a is a
    // constraint residual whose natural scale is the magnitude of the
    // φ_a rows (O(1)–O(10) in the h_chart=1-rescaled chart). The 5-dim
    // SME dual is more poorly conditioned than the 2-dim LME dual (its
    // μ rows are products D_a = u⊗u − d_a), so g cannot be driven to an
    // ABSOLUTE 1e-10 in double precision — it floors at the conditioning
    // limit ~1e-10·cond(H). Testing |g|_∞ ≤ newton_tol absolutely then
    // never trips and a feasible, converged solve is mis-reported as a
    // divergence. Scale the tolerance by the φ magnitude so the test is
    // relative (matches Arroyo–Ortiz LME practice); the LME path is
    // better conditioned and hits the same scaled bar trivially.
    const double phi_scale =
        std::max(1.0, s.PHI.cwiseAbs().maxCoeff());
    const double g_tol = newton_tol * phi_scale;

    Eigen::VectorXd g(5);
    Eigen::MatrixXd H_clean(5, 5);
    int iter = 0;
    for (; iter < newton_max_iters; ++iter) {
        g = s.PHI.transpose() * s.p;
        if (g.lpNorm<Eigen::Infinity>() <= g_tol) {
            // Stash the unridged Hessian for the IFT path before
            // breaking; downstream gradient/Hessian solvers want the
            // exact Fisher information, not the ridged solver matrix.
            H_clean = s.PHI.transpose() * s.p.asDiagonal() * s.PHI
                      - g * g.transpose();
            break;
        }

        H_clean = s.PHI.transpose() * s.p.asDiagonal() * s.PHI
                  - g * g.transpose();
        const double H_trace = H_clean.trace();
        // Gradient-scaled Tikhonov ridge. Far from the solution
        // (g ~ O(1)) the ridge is ~1e-8·tr H, stabilising the near-
        // singular Hessian at chart-boundary / near-node queries. As
        // Newton approaches the minimiser the ridge shrinks with the
        // gradient and vanishes, restoring the UNridged Hessian and
        // hence quadratic convergence to machine precision. A FIXED
        // 1e-8·tr H ridge instead floors the achievable gradient at
        // ~1e-9 (the ridged step can no longer reduce a gradient below
        // the ridge), so the |g| ≤ newton_tol (1e-10) test never trips
        // and a feasible, essentially-converged solve is mis-reported
        // as a divergence — the SME α=2 "divergence" on curved meshes.
        // The conditioning floor is α-dependent (smaller gaps ⇒ worse
        // conditioning), which is why a fixed ridge let α=4 through but
        // not the paper's α=2.
        const double g_inf  = g.lpNorm<Eigen::Infinity>();
        const double ridge  =
            std::max(1.0e-14, 1.0e-8 * H_trace * std::min(1.0, g_inf));
        Eigen::MatrixXd H_solver = H_clean +
            ridge * Eigen::MatrixXd::Identity(5, 5);

        Eigen::VectorXd dtheta = H_solver.ldlt().solve(g);
        const double dtheta_norm = dtheta.norm();
        if (dtheta_norm > kMaxDthetaNorm) {
            dtheta *= kMaxDthetaNorm / dtheta_norm;
        }
        const double g_dot_dtheta = g.dot(dtheta);

        double step   = 1.0;
        Eigen::VectorXd theta_trial = s.theta;
        Eigen::VectorXd p_trial(n_act);
        double f_trial = f_curr;
        bool   armijo_ok = false;
        while (step > kMinStep) {
            theta_trial = s.theta - step * dtheta;
            f_trial = log_sum_exp_minus_logZ(theta_trial, p_trial);
            if (f_trial <= f_curr - kArmijoC1 * step * g_dot_dtheta) {
                armijo_ok = true;
                break;
            }
            step *= 0.5;
        }
        // Objective-stagnation convergence. The line search either found
        // no decreasing step (!armijo_ok) or the achieved decrease has
        // fallen to the floating-point floor (df ≈ 0): for the strictly-
        // convex SME dual both mean the current iterate is the minimiser
        // to numerical precision — converged — even though |g| still sits
        // at the Hessian-conditioning floor (~1e-8 on poorly-conditioned
        // wPCA-projected curved charts, far above any achievable absolute
        // newton_tol). This is the crux of the α=2 "divergence" fix: the
        // dual is feasible (0 ∈ int conv{φ_a}) and fully solved (|dθ|~1e-8,
        // df=0), but |g| exceeds the tolerance so the |g| ≤ tol test never
        // trips and Newton spins to max-iters. The decisive symptom was the
        // Armijo required-decrease c1·step·(g·dθ) underflowing to 0, making
        // every microscopic step "succeed" with df=0.
        //
        // This does NOT mask a genuinely INFEASIBLE program: there ln Z is
        // unbounded below (θ runs along a recession ray), every iterate
        // achieves a substantial, non-vanishing decrease (df = O(1e-2)), so
        // neither branch fires and the solve still exits via the max-iters
        // throw below — the correct infeasibility signal.
        const double df = f_curr - f_trial;   // ≥ 0 when armijo_ok
        constexpr double kConvergeFtol = 1.0e-13;
        if (!armijo_ok ||
            df <= kConvergeFtol * (1.0 + std::abs(f_curr)))
        {
            if (armijo_ok) { s.theta = theta_trial; s.p = p_trial; }
            H_clean = s.PHI.transpose() * s.p.asDiagonal() * s.PHI
                      - g * g.transpose();
            s.H = H_clean;
            return s;
        }
        s.theta = theta_trial;
        s.p     = p_trial;
        f_curr  = f_trial;
    }
    if (iter == newton_max_iters) {
        if (std::getenv("SME_DIAG")) {
            // Recession-direction test: if Newton's θ runs to infinity
            // along a ray v with v·φ_a < 0 ∀a, then ln Z is unbounded
            // below ⇒ 0 ∉ int conv{φ_a} ⇒ the SME program is truly
            // INFEASIBLE at x (no slack/iterate fixes it). Otherwise the
            // failure is a Newton/conditioning issue on a feasible
            // program. maxproj < 0 confirms a separating hyperplane.
            const double tn = s.theta.norm();
            const Eigen::VectorXd vdir = s.theta / (tn + 1e-30);
            const Eigen::VectorXd proj = s.PHI * vdir;  // v·φ_a per node
            int n_zero_d = 0;
            for (const auto& da : d)
                if (da.cwiseAbs().maxCoeff() < kSmeZeroDTol) ++n_zero_d;
            std::fprintf(stderr,
                "  [sme_newton_fail] n_act=%lld n_zero_d(all)=%d "
                "||theta||=%.3e gNorm=%.3e maxproj(v.phi)=%.3e "
                "%s\n",
                (long long)n_act, n_zero_d, tn,
                g.lpNorm<Eigen::Infinity>(), proj.maxCoeff(),
                proj.maxCoeff() < 0 ? "INFEASIBLE(separating hyperplane)"
                                    : "feasible? (Newton/cond issue)");
        }
        throw std::runtime_error(
            std::string("lme::") + caller +
            ": Newton failed to converge in " +
            std::to_string(newton_max_iters) + " iterations " +
            "(tolerance " + std::to_string(newton_tol) + "). " +
            "Typically indicates that x is outside the feasibility " +
            "region — check that the d_a were built via " +
            "compute_nodal_gaps (or an equivalent feasible recipe)");
    }

    s.H = H_clean;
    return s;
}

/// Build LMEBasisAndGrad from a converged SMEState. Factored out of
/// @ref evaluate_sme_basis_and_grad so the gradient-assembly logic
/// can be reused inside @ref evaluate_sme_basis_grad_and_hess
/// without re-running the Newton solve (the FD-Hessian path needs
/// per-perturbed-query gradients but warm-starts Newton on the
/// converged at-x θ — see that function's comments).
LMEBasisAndGrad build_grad_from_state(
    const SMEState&                              s,
    const char*                                  caller)
{
    if (s.exact_match) {
        throw std::domain_error(
            std::string("lme::") + caller +
            ": query x coincides with node " +
            std::to_string(s.exact_match_id) +
            " (boundary corner with d_b = 0). The SME basis is not "
            "differentiable at such nodes — same as 1st-order LME at "
            "conv-hull corners.");
    }

    const Eigen::Index n_act =
        static_cast<Eigen::Index>(s.active.size());

    // Per-node 5x2 Jacobian J_a = ∂φ_a/∂x.
    std::vector<Eigen::Matrix<double, 5, 2>> JAC(
        static_cast<std::size_t>(n_act));
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const double u1 = s.PHI(k, 0);
        const double u2 = s.PHI(k, 1);
        Eigen::Matrix<double, 5, 2>& J =
            JAC[static_cast<std::size_t>(k)];
        J << 1.0,        0.0,
             0.0,        1.0,
             -2.0 * u1,  0.0,
             0.0,        -2.0 * u2,
             -2.0 * u2,  -2.0 * u1;
    }

    // IFT right-hand side G ∈ R^{5x2}.
    Eigen::MatrixXd R(n_act, 2);   // R[a, k] = θ* · J_a[:, k]
    Eigen::Matrix<double, 5, 2> J_avg = Eigen::Matrix<double, 5, 2>::Zero();
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const Eigen::Matrix<double, 5, 2>& J =
            JAC[static_cast<std::size_t>(k)];
        R.row(k)  = (J.transpose() * s.theta).transpose();
        J_avg    += s.p(k) * J;
    }
    Eigen::MatrixXd G(5, 2);
    G = s.PHI.transpose() * s.p.asDiagonal() * R + J_avg;

    // Solve H · ∂θ*/∂x = -G.
    Eigen::MatrixXd H_ift = s.H;
    const double H_trace = H_ift.trace();
    const double ridge   = std::max(1.0e-14, 1.0e-10 * H_trace);
    H_ift += ridge * Eigen::MatrixXd::Identity(5, 5);
    const Eigen::MatrixXd dtheta_dx = H_ift.ldlt().solve(-G);  // 5x2

    // Per-node gradient.
    const Eigen::RowVector2d r_avg = s.p.transpose() * R;

    LMEBasisAndGrad out;
    out.indices.resize(static_cast<std::size_t>(n_act));
    out.values.resize(static_cast<std::size_t>(n_act));
    out.gradients.resize(static_cast<std::size_t>(n_act));
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const std::size_t kz = static_cast<std::size_t>(k);
        out.indices[kz] = s.active[kz];
        out.values[kz]  = s.p(k);

        const Eigen::RowVector2d r_k = R.row(k);
        const Eigen::RowVector2d dtheta_phi =
            s.PHI.row(k) * dtheta_dx;
        const Eigen::RowVector2d grad_row =
            s.p(k) * (dtheta_phi + r_k - r_avg);
        out.gradients[kz] = grad_row.transpose();
    }
    return out;
}

/// Closed-form SME basis gradient **and** Hessian from a converged
/// state — the analytic replacement for the FD-on-gradient path in
/// @ref evaluate_sme_basis_grad_and_hess. Follows Rosolen-Millán-Arroyo
/// 2013 Appendix C (gradient Eq. C.3, Hessian Eq. C.4) but derived
/// directly in this file's θ = (λ_1, λ_2, μ_11, μ_22, μ_12) packing so
/// the paper's Voigt factor-of-2 placement (its φ off-diagonal carries
/// no 2; ours bakes the 2 into φ_a[4] = −2 D_a,12) needs no separate
/// bookkeeping.
///
/// Derivation (all sums over the active set, evaluated at the converged
/// θ* where Σ_a p_a φ_a = 0, so the softmax mean φ̄ = 0):
///
///   Let u_a = x − x_a, φ_a the 5-vector packed in @ref compute_sme_state,
///   J_a = ∂φ_a/∂x (5×2, the @c JAC below), and Θ = ∂θ*/∂x (5×2, the
///   @c dtheta_dx from the first IFT H·Θ = −G). Write the *total* spatial
///   gradient of the log-weight f_a = θ*ᵀφ_a as
///       ∇f_a = Θᵀφ_a + J_aᵀθ*   (∈ R²),
///   and g_a := ∇f_a − Σ_b p_b ∇f_b, so ∇p_a = p_a g_a (already what
///   @ref build_grad_from_state returns). At θ* the mean of the Θᵀφ part
///   vanishes, so g_a = ψ_a + δ_a with ψ_a = Θᵀφ_a and δ_a = R_a − R̄.
///
///   The Hessian (Eq. C.4, with the constant −2μ̄ term of Hf_a dropping
///   out of the mean-subtracted combination) is
///       ∇²p_a = p_a [ g_a g_aᵀ − Σ_b p_b g_b g_bᵀ
///                     + Hf̃_a − Σ_b p_b Hf̃_b ],
///   where the mean-free log-weight Hessian is
///       (Hf̃_a)_ij = Σ_ij·φ_a + (P_a)_ij + (P_a)_ji,
///   P_a = J_aᵀΘ (2×2), and Σ_ij = ∂²θ*/∂x_i∂x_j ∈ R⁵ comes from a
///   *second* implicit-function-theorem solve against the same Fisher
///   information H:
///       H·Σ_ij = −[ Σ_a p_a (g_a,i g_a,j + (P_a)_ij + (P_a)_ji) φ_a
///                   + Σ_a p_a g_a,j (J_a)_:,i
///                   + Σ_a p_a g_a,i (J_a)_:,j
///                   + κ_ij ],
///   with κ_ij = Σ_a p_a ∂²φ_a/∂x_i∂x_j a node-independent constant:
///   κ_00 = −2 e_2, κ_11 = −2 e_3, κ_01 = −2 e_4 (0-based φ indices,
///   from φ_a[2..4] = (−D_11, −D_22, −2D_12)).
///
/// The 1/s_a·∇s_a factor the paper warns about (its App-C remark on
/// error amplification as s_a → 0) never appears here: g_a is built
/// additively, never by dividing a gradient by p_a.
LMEBasisGradHess build_grad_and_hess_from_state(
    const SMEState&                              s,
    const char*                                  caller)
{
    if (s.exact_match) {
        throw std::domain_error(
            std::string("lme::") + caller +
            ": query x coincides with node " +
            std::to_string(s.exact_match_id) +
            " (boundary corner with d_b = 0). The SME basis is not "
            "differentiable at such nodes — same as 1st-order LME at "
            "conv-hull corners.");
    }

    const Eigen::Index n_act =
        static_cast<Eigen::Index>(s.active.size());

    // ---- gradient stage (mirrors build_grad_from_state) -------------
    std::vector<Eigen::Matrix<double, 5, 2>> JAC(
        static_cast<std::size_t>(n_act));
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const double u1 = s.PHI(k, 0);
        const double u2 = s.PHI(k, 1);
        Eigen::Matrix<double, 5, 2>& J =
            JAC[static_cast<std::size_t>(k)];
        J << 1.0,        0.0,
             0.0,        1.0,
             -2.0 * u1,  0.0,
             0.0,        -2.0 * u2,
             -2.0 * u2,  -2.0 * u1;
    }

    Eigen::MatrixXd R(n_act, 2);   // R[a, k] = θ* · J_a[:, k]
    Eigen::Matrix<double, 5, 2> J_avg = Eigen::Matrix<double, 5, 2>::Zero();
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const Eigen::Matrix<double, 5, 2>& J =
            JAC[static_cast<std::size_t>(k)];
        R.row(k)  = (J.transpose() * s.theta).transpose();
        J_avg    += s.p(k) * J;
    }
    Eigen::MatrixXd G(5, 2);
    G = s.PHI.transpose() * s.p.asDiagonal() * R + J_avg;

    // Factorise the (ridged) Fisher information once and reuse it for
    // both the first IFT (Θ = ∂θ*/∂x) and the three second-IFT solves
    // for Σ_ij — the ridge keeps both consistent so the closed form is
    // the exact derivative of build_grad_from_state's ridged gradient.
    Eigen::MatrixXd H_ift = s.H;
    const double H_trace = H_ift.trace();
    const double ridge   = std::max(1.0e-14, 1.0e-10 * H_trace);
    H_ift += ridge * Eigen::MatrixXd::Identity(5, 5);
    const Eigen::LDLT<Eigen::MatrixXd> H_ldlt = H_ift.ldlt();
    const Eigen::MatrixXd dtheta_dx = H_ldlt.solve(-G);  // 5x2

    const Eigen::RowVector2d r_avg = s.p.transpose() * R;

    LMEBasisGradHess out;
    out.indices.resize(static_cast<std::size_t>(n_act));
    out.values.resize(static_cast<std::size_t>(n_act));
    out.gradients.resize(static_cast<std::size_t>(n_act));
    out.hessians.resize(static_cast<std::size_t>(n_act));

    // Per-node g_a (= ∇p_a / p_a, built additively) and P_a = J_aᵀΘ.
    std::vector<Eigen::Vector2d> gvec(static_cast<std::size_t>(n_act));
    std::vector<Eigen::Matrix2d> Pmat(static_cast<std::size_t>(n_act));
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const std::size_t kz = static_cast<std::size_t>(k);
        out.indices[kz] = s.active[kz];
        out.values[kz]  = s.p(k);

        const Eigen::RowVector2d dtheta_phi = s.PHI.row(k) * dtheta_dx;
        const Eigen::RowVector2d g_row =
            dtheta_phi + R.row(k) - r_avg;        // = g_a (row)
        gvec[kz] = g_row.transpose();
        out.gradients[kz] = (s.p(k) * g_row).transpose();
        Pmat[kz] = JAC[kz].transpose() * dtheta_dx;  // 2x2
    }

    // ---- second IFT: solve H·Σ_ij = −RHS_ij for the three unique
    //      symmetric spatial pairs (0,0), (1,1), (0,1) ----------------
    // RHS columns are ordered [ (0,0), (1,1), (0,1) ].
    constexpr int pair_i[3] = {0, 1, 0};
    constexpr int pair_j[3] = {0, 1, 1};
    constexpr int kappa_phi_idx[3] = {2, 3, 4};  // φ index carrying −2

    Eigen::MatrixXd RHS(5, 3);
    for (int p = 0; p < 3; ++p) {
        const int i = pair_i[p];
        const int j = pair_j[p];
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(5);
        for (Eigen::Index k = 0; k < n_act; ++k) {
            const std::size_t kz = static_cast<std::size_t>(k);
            const Eigen::Vector2d& g = gvec[kz];
            const Eigen::Matrix2d& P = Pmat[kz];
            const double pk = s.p(k);
            const double w  =
                g(i) * g(j) + P(i, j) + P(j, i);
            rhs += pk * w * s.PHI.row(k).transpose();
            rhs += pk * g(j) * JAC[kz].col(i);
            rhs += pk * g(i) * JAC[kz].col(j);
        }
        rhs(kappa_phi_idx[p]) += -2.0;   // κ_ij
        RHS.col(p) = rhs;
    }
    const Eigen::MatrixXd Sigma = H_ldlt.solve(-RHS);  // 5x3

    // ---- assemble Eq. C.4 -------------------------------------------
    std::vector<Eigen::Matrix2d> Hf(static_cast<std::size_t>(n_act));
    Eigen::Matrix2d barGG = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d barHf = Eigen::Matrix2d::Zero();
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const std::size_t kz = static_cast<std::size_t>(k);
        const Eigen::Matrix2d& P = Pmat[kz];
        const double s00 = (s.PHI.row(k) * Sigma.col(0)).value();
        const double s11 = (s.PHI.row(k) * Sigma.col(1)).value();
        const double s01 = (s.PHI.row(k) * Sigma.col(2)).value();

        Eigen::Matrix2d Hf_a;
        Hf_a(0, 0) = s00 + 2.0 * P(0, 0);
        Hf_a(1, 1) = s11 + 2.0 * P(1, 1);
        const double off = s01 + P(0, 1) + P(1, 0);
        Hf_a(0, 1) = off;
        Hf_a(1, 0) = off;
        Hf[kz] = Hf_a;

        barGG += s.p(k) * gvec[kz] * gvec[kz].transpose();
        barHf += s.p(k) * Hf_a;
    }

    for (Eigen::Index k = 0; k < n_act; ++k) {
        const std::size_t kz = static_cast<std::size_t>(k);
        const Eigen::Matrix2d Hk =
            gvec[kz] * gvec[kz].transpose() - barGG
            + Hf[kz] - barHf;
        out.hessians[kz] = s.p(k) * Hk;
    }
    return out;
}

}  // namespace

LMEBasisValues evaluate_sme_basis(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    const SMEState s = compute_sme_state(
        nodes, d, x, r_cut, newton_tol, newton_max_iters,
        "evaluate_sme_basis");

    LMEBasisValues bv;
    if (s.exact_match) {
        bv.indices = { s.exact_match_id };
        bv.values  = { 1.0 };
        return bv;
    }

    const std::size_t n_act = s.active.size();
    bv.indices.resize(n_act);
    bv.values.resize(n_act);
    for (std::size_t k = 0; k < n_act; ++k) {
        bv.indices[k] = s.active[k];
        bv.values[k]  = s.p(static_cast<Eigen::Index>(k));
    }
    return bv;
}

LMEBasisAndGrad evaluate_sme_basis_and_grad(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    const SMEState s = compute_sme_state(
        nodes, d, x, r_cut, newton_tol, newton_max_iters,
        "evaluate_sme_basis_and_grad");
    return build_grad_from_state(s, "evaluate_sme_basis_and_grad");
}

LMEBasisGradHess evaluate_sme_basis_grad_and_hess(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    // Cold-start Newton at x and keep the converged state so the 4
    // FD-perturbed queries below can warm-start from the same θ.
    // At h=1e-5 the perturbed queries' optimal θ is within ~h·|σ|
    // of θ*(x), and Newton converges in 1-2 iters from there instead
    // of the 8-15 cold-start typically needed. Measured 3x SME K
    // speedup on the polar disk fixture.
    const SMEState s_at_x = compute_sme_state(
        nodes, d, x, r_cut, newton_tol, newton_max_iters,
        "evaluate_sme_basis_grad_and_hess");
    const auto at_x = build_grad_from_state(
        s_at_x, "evaluate_sme_basis_grad_and_hess");
    const Eigen::VectorXd theta_x = s_at_x.theta;  // 5-vec capture

    // Central FD on the gradient. Step h chosen so that the O(h²)
    // truncation residual (~h² · ‖∇³s‖) is well below the Newton
    // convergence floor (~1e-10): h = 1e-5 gives ~1e-10 absolute on
    // smooth interior bases. Pushing h smaller hits Newton noise.
    constexpr double kFdStep = 1.0e-5;

    Eigen::Vector2d xv(2);
    xv << x(0), x(1);

    auto grad_at_warm = [&](const Eigen::Vector2d& xq) {
        const SMEState s = compute_sme_state(
            nodes, d, xq, r_cut, newton_tol, newton_max_iters,
            "evaluate_sme_basis_grad_and_hess", &theta_x);
        return build_grad_from_state(
            s, "evaluate_sme_basis_grad_and_hess");
    };

    const auto gp_x = grad_at_warm(xv + Eigen::Vector2d(kFdStep, 0.0));
    const auto gm_x = grad_at_warm(xv - Eigen::Vector2d(kFdStep, 0.0));
    const auto gp_y = grad_at_warm(xv + Eigen::Vector2d(0.0, kFdStep));
    const auto gm_y = grad_at_warm(xv - Eigen::Vector2d(0.0, kFdStep));

    // Active-set sanity check — if the FD-perturbed queries see a
    // different active set than at_x, the Hessian we produce is
    // garbage. Callers in Galerkin assembly must pick r_cut so this
    // never trips; expose the failure loudly during development.
    auto same_indices = [&](const std::vector<int>& a,
                             const std::vector<int>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    };
    if (!same_indices(at_x.indices, gp_x.indices) ||
        !same_indices(at_x.indices, gm_x.indices) ||
        !same_indices(at_x.indices, gp_y.indices) ||
        !same_indices(at_x.indices, gm_y.indices))
    {
        throw std::runtime_error(
            "lme::evaluate_sme_basis_grad_and_hess: the active set "
            "changes between x and the FD-perturbed queries — increase "
            "r_cut so all of x ± h e_l (h = 1e-5) see the same "
            "neighbour list");
    }

    const std::size_t n_act = at_x.indices.size();

    LMEBasisGradHess out;
    out.indices  = at_x.indices;
    out.values   = at_x.values;
    out.gradients = at_x.gradients;
    out.hessians.resize(n_act);

    for (std::size_t k = 0; k < n_act; ++k) {
        const Eigen::Vector2d col_x =
            (gp_x.gradients[k] - gm_x.gradients[k]) / (2.0 * kFdStep);
        const Eigen::Vector2d col_y =
            (gp_y.gradients[k] - gm_y.gradients[k]) / (2.0 * kFdStep);

        Eigen::Matrix2d H_fd;
        H_fd.col(0) = col_x;
        H_fd.col(1) = col_y;
        // Symmetrise: central FD on the gradient is symmetric only up
        // to the FD truncation residual, but the Hessian semantically
        // is symmetric, so average with the transpose.
        out.hessians[k] = 0.5 * (H_fd + H_fd.transpose().eval());
    }
    return out;
}

LMEBasisGradHess evaluate_sme_basis_grad_and_hess_closed_form(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    // Closed-form path (Rosolen-Millán-Arroyo 2013 App. C): one Newton
    // solve at x, then the gradient + Hessian via two implicit-function-
    // theorem passes against the converged Fisher information — no FD,
    // no extra perturbed Newton solves, and no active-set sensitivity.
    // See @ref build_grad_and_hess_from_state for the per-node formula.
    const SMEState s = compute_sme_state(
        nodes, d, x, r_cut, newton_tol, newton_max_iters,
        "evaluate_sme_basis_grad_and_hess_closed_form");
    return build_grad_and_hess_from_state(
        s, "evaluate_sme_basis_grad_and_hess_closed_form");
}

namespace {

/// Tolerance for the "x coincides with node x_b" exact-match shortcut.
/// Absolute squared distance threshold; comfortably below the smallest
/// inter-node spacing we expect in practice (~1e-3 for fine meshes).
constexpr double kExactNodeMatchEps2 = 1.0e-28;

/// Newton-loop state shared by the public entry points.
/// On exact-node match @c exact_match is true and the other fields
/// (other than @c exact_match_id) are left empty.
struct LMEState {
    bool             exact_match     = false;
    int              exact_match_id  = -1;
    std::vector<int> active;       ///< Node ids in the active set
    Eigen::MatrixXd  M;            ///< n_act x d  (rows are m_a = x - x_a)
    Eigen::VectorXd  p;            ///< n_act       (basis values)
    Eigen::MatrixXd  J;            ///< d x d       (converged Fisher info, no ridge)
};

/// Common preflight + Newton solve. Throws on invalid inputs or
/// non-convergence; on success returns a fully-populated @ref LMEState
/// (with @c exact_match = true at node coincidence).
LMEState compute_state(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    // ----- input validation -------------------------------------------
    if (nodes.rows() == 0) {
        throw std::invalid_argument(
            "lme::evaluate_basis: nodes matrix is empty");
    }
    if (beta.size() != nodes.rows()) {
        throw std::invalid_argument(
            "lme::evaluate_basis: beta.size() must equal nodes.rows()");
    }
    if (x.size() != nodes.cols()) {
        throw std::invalid_argument(
            "lme::evaluate_basis: x dimension must equal nodes.cols()");
    }
    for (Eigen::Index a = 0; a < beta.size(); ++a) {
        if (!(beta(a) > 0.0)) {
            throw std::invalid_argument(
                "lme::evaluate_basis: every beta_a must be strictly positive");
        }
    }
    if (!(r_cut > 0.0)) {
        throw std::invalid_argument(
            "lme::evaluate_basis: r_cut must be strictly positive");
    }

    const Eigen::Index n_nodes = nodes.rows();
    const Eigen::Index d       = nodes.cols();
    const double       r_cut2  = r_cut * r_cut;

    // ----- active set: nodes within r_cut, plus exact-match check ----
    LMEState s;
    s.active.reserve(static_cast<std::size_t>(n_nodes));
    for (Eigen::Index a = 0; a < n_nodes; ++a) {
        const double d2 = (x.transpose() - nodes.row(a)).squaredNorm();
        if (d2 <= kExactNodeMatchEps2) {
            s.exact_match    = true;
            s.exact_match_id = static_cast<int>(a);
            return s;
        }
        if (d2 <= r_cut2) {
            s.active.push_back(static_cast<int>(a));
        }
    }
    if (s.active.empty()) {
        throw std::invalid_argument(
            "lme::evaluate_basis: no nodes within r_cut of x — "
            "increase r_cut or check that x lies inside the node cloud");
    }

    const Eigen::Index n_act = static_cast<Eigen::Index>(s.active.size());

    // Pre-stage the active-set offsets m_a = x - x_a and the constant
    // log-weight contribution -β_a |m_a|² (the Newton iteration only
    // varies the λ·m_a piece on top).
    s.M.resize(n_act, d);
    Eigen::VectorXd f_const(n_act);
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const int a = s.active[static_cast<std::size_t>(k)];
        s.M.row(k)  = x.transpose() - nodes.row(a);
        f_const(k)  = -beta(a) * s.M.row(k).squaredNorm();
    }

    // ----- Newton iteration on λ in R^d ------------------------------
    // Undamped Newton on the convex dual log-partition
    //   F(λ) = log Σ_a exp(f_const_a + λ·m_a),  f_const_a = -β_a |m_a|²,
    // whose stationary point g = ∇F = Σ p_a m_a = 0 is the LME constraint
    // (the reconstruction residual x − Σ p_a x_a).
    //
    // R1 NOTE (review fragility item, 2026-06-09): the review flagged this
    // as fragile relative to compute_sme_state — bare undamped Newton, a
    // fixed 1e-14 ridge, and an ABSOLUTE |g| ≤ newton_tol test rather than
    // SME's gradient-scaled ridge + Armijo line search + relative tol. We
    // tried porting that machinery here and it REGRESSED 20+ LME fixtures:
    // the 2-D LME dual is well-conditioned and the full Newton step already
    // converges quadratically, so adding Armijo damping only slowed it below
    // the iteration budget on the harder charts, cascading into drop-net
    // breakdown. SME needs the damping because its 5-D dual is genuinely
    // ill-conditioned; the 2-D LME dual does not. The undamped solver is
    // empirically robust on every shipped fixture, and the assembler's
    // k-ring drop-net (see [sme_no_drops]) is the designed backstop for the
    // rare pathological chart. Kept undamped by measurement, not oversight.
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(d);
    s.p.resize(n_act);

    int iter = 0;
    for (; iter < newton_max_iters; ++iter) {
        // f_k = -β_a |m_a|² + λ·m_a  with log-sum-exp shift.
        Eigen::VectorXd f = f_const + s.M * lambda;
        const double f_max = f.maxCoeff();
        f.array() -= f_max;
        s.p = f.array().exp();
        const double Z = s.p.sum();
        s.p /= Z;

        // Gradient ∇F = Σ p_a m_a.  (Equals -(reconstruction error)
        // since Σ p_a m_a = Σ p_a x - Σ p_a x_a = x - Σ p_a x_a.)
        const Eigen::VectorXd g = s.M.transpose() * s.p;
        if (g.lpNorm<Eigen::Infinity>() <= newton_tol) {
            break;
        }

        // Hessian ∇²F = Σ p_a m_a m_a^T - g g^T  (Fisher information).
        // For d=2 (current use) Eigen's direct LDLT is overkill — we
        // just solve the dxd system via inverse(), which Eigen
        // specialises for d<=4. Add a tiny ridge to stay PD near the
        // conv-hull boundary where the Hessian can be nearly singular
        // along the inward direction; ridge magnitude is well below
        // newton_tol's impact on the dual gradient.
        Eigen::MatrixXd H = s.M.transpose() * s.p.asDiagonal() * s.M
                            - g * g.transpose();
        H += 1e-14 * Eigen::MatrixXd::Identity(d, d);

        const Eigen::VectorXd dlambda = H.ldlt().solve(g);
        lambda -= dlambda;
    }
    if (iter == newton_max_iters) {
        throw std::runtime_error(
            "lme::evaluate_basis: Newton failed to converge in "
            + std::to_string(newton_max_iters) + " iterations "
            "(tolerance " + std::to_string(newton_tol) + ")");
    }

    // Cache the (unregularised) Fisher info at λ*: needed downstream
    // by the gradient evaluator. At convergence g ≈ 0 so the g·g^T
    // correction is below newton_tol² and we drop it.
    s.J = s.M.transpose() * s.p.asDiagonal() * s.M;
    return s;
}

}  // namespace

LMEBasisValues evaluate_basis(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    const auto s = compute_state(nodes, beta, x, r_cut,
                                  newton_tol, newton_max_iters);

    LMEBasisValues bv;
    if (s.exact_match) {
        bv.indices = { s.exact_match_id };
        bv.values  = { 1.0 };
        return bv;
    }
    bv.indices = s.active;
    bv.values.assign(s.p.data(), s.p.data() + s.p.size());
    return bv;
}

LMEBasisAndGrad evaluate_basis_and_grad(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    auto s = compute_state(nodes, beta, x, r_cut,
                            newton_tol, newton_max_iters);

    if (s.exact_match) {
        throw std::domain_error(
            "lme::evaluate_basis_and_grad: gradient is not well-defined "
            "at node coincidence — the LME basis has a kink at conv-hull "
            "corners (Hessian J degenerates)");
    }

    // Closed-form gradients per Millán 2011 Appendix A (A5)-(A6) for
    // PER-NODE β_a — the form the assemblers actually need, since they
    // pass β_a = γ/h_a² with the nodal spacing h_a:
    //     ∇p_a = p_a [ r̄ − M_a m_a ],      m_a = x − x_a,
    //     r̄   = 2 Σ_b β_b p_b m_b,
    //     J̄   = 2 Σ_b β_b p_b m_b ⊗ m_b,
    //     M_a  = 2 β_a I − Dk*,   Dk* = (J̄ − I)(J*)^{-1},   J* = J.
    // Computed as g_a = r̄ − 2 β_a m_a + (J̄ − I) v_a with
    // v_a = J^{-1} m_a, so the uniform-β limit collapses exactly to
    // the Arroyo-Ortiz 2006 eq. 44 form −p_a J^{-1} m_a (r̄ → 0 and
    // (2βJ − I)J^{-1} m_a = 2β m_a − v_a). HISTORY (2026-06-03): the
    // previous code used that uniform-β special case unconditionally;
    // on node sets with strong h_a modulation (QuadSplit UnionJack /
    // Checkerboard sublattices) the missing r̄ / J̄ terms made every
    // assembled derivative wrong by O(β-spread) — the root cause of
    // the spurious sub-spectrum m=0 mode on those meshes.
    const Eigen::Index d_g   = s.M.cols();
    const Eigen::Index n_act = static_cast<Eigen::Index>(s.active.size());
    Eigen::VectorXd pb(n_act);
    for (Eigen::Index k = 0; k < n_act; ++k) {
        pb(k) = s.p(k) * beta(s.active[static_cast<std::size_t>(k)]);
    }
    const Eigen::VectorXd r_bar = 2.0 * s.M.transpose() * pb;
    const Eigen::MatrixXd Jbar =
        2.0 * s.M.transpose() * pb.asDiagonal() * s.M;
    const Eigen::MatrixXd V = s.J.ldlt().solve(s.M.transpose());
    const Eigen::MatrixXd JbV =
        (Jbar - Eigen::MatrixXd::Identity(d_g, d_g)) * V;

    LMEBasisAndGrad bg;
    bg.values.assign(s.p.data(), s.p.data() + s.p.size());
    bg.gradients.reserve(static_cast<std::size_t>(n_act));
    for (Eigen::Index k = 0; k < n_act; ++k) {
        const double beta_k = beta(s.active[static_cast<std::size_t>(k)]);
        bg.gradients.emplace_back(
            s.p(k) * (r_bar - 2.0 * beta_k * s.M.row(k).transpose()
                      + JbV.col(k)));
    }
    bg.indices = std::move(s.active);
    return bg;
}

LMEBasisGradHess evaluate_basis_grad_and_hess(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    auto s = compute_state(nodes, beta, x, r_cut,
                            newton_tol, newton_max_iters);

    if (s.exact_match) {
        throw std::domain_error(
            "lme::evaluate_basis_grad_and_hess: gradient/Hessian are "
            "not well-defined at node coincidence — the LME basis has a "
            "kink at conv-hull corners (Hessian J degenerates)");
    }

    const Eigen::Index d     = s.M.cols();
    const Eigen::Index n_act = static_cast<Eigen::Index>(s.active.size());

    // Shared geometry — same J^{-1} m_a column matrix as in
    // evaluate_basis_and_grad.
    Eigen::LDLT<Eigen::MatrixXd> Jldlt = s.J.ldlt();
    Eigen::MatrixXd V    = Jldlt.solve(s.M.transpose());   // d x n_act, V[:,a] = v_a

    // α_{ba} = m_b · v_a = m_b·(J*)^{-1} m_a — the paper's κ_ab
    // (Millán 2011 App A). Row b holds m_b's dot products against
    // every v_a.
    Eigen::MatrixXd alpha = s.M * V;  // n_act × n_act

    // ----- Millán 2011 Appendix A, per-node β_a (A5)-(A7) ------------
    // Gradient:  ∇p_a = p_a g_a,
    //   g_a = r̄ − M_a m_a = r̄ − 2 β_a m_a + (J̄ − I) v_a,
    //   r̄  = 2 Σ_b β_b p_b m_b,    J̄ = 2 Σ_b β_b p_b m_b ⊗ m_b.
    // Hessian (A7):
    //   ∇²p_a = p_a [ g_a ⊗ g_a + 2(β̄ − β_a) I
    //                + r̄⊗r̄ + r̄⊗j_a + j_a⊗r̄ + (r̄·j_a) I
    //                − Σ_b p_b (1 + κ_ab) (M_b m_b) ⊗ (M_b m_b) ],
    // with β̄ = Σ_b β_b p_b, j_a = v_a, and M_b m_b = r̄ − g_b. The
    // uniform-β limit collapses exactly to the previous AO06 eq. 44
    // closed form (r̄ → 0, g_a → −v_a, M_b m_b → v_b, the (1+κ) sum
    // → second_term + J^{-1}). HISTORY (2026-06-03): the uniform-β
    // special case was previously applied unconditionally; on
    // h_a-modulated node sets (QuadSplit sublattices) the missing
    // terms made the assembled bending energy non-variational — the
    // spurious sub-spectrum m=0 mode's root cause.
    Eigen::VectorXd beta_act(n_act);
    for (Eigen::Index k = 0; k < n_act; ++k) {
        beta_act(k) = beta(s.active[static_cast<std::size_t>(k)]);
    }
    const Eigen::VectorXd pb     = s.p.array() * beta_act.array();
    const double          b_mean = pb.sum();              // β̄
    const Eigen::VectorXd r_bar  = 2.0 * s.M.transpose() * pb;
    const Eigen::MatrixXd Jbar =
        2.0 * s.M.transpose() * pb.asDiagonal() * s.M;
    const Eigen::MatrixXd JbV =
        (Jbar - Eigen::MatrixXd::Identity(d, d)) * V;

    // g_a columns and w_b = M_b m_b = r̄ − g_b columns.
    Eigen::MatrixXd G(d, n_act);
    for (Eigen::Index k = 0; k < n_act; ++k) {
        G.col(k) = r_bar - 2.0 * beta_act(k) * s.M.row(k).transpose()
                   + JbV.col(k);
    }
    const Eigen::MatrixXd W = r_bar.replicate(1, n_act) - G;

    LMEBasisGradHess gh;
    gh.values.assign(s.p.data(), s.p.data() + s.p.size());
    gh.gradients.reserve(static_cast<std::size_t>(n_act));
    gh.hessians.reserve(static_cast<std::size_t>(n_act));

    const Eigen::MatrixXd r_outer = r_bar * r_bar.transpose();
    for (Eigen::Index a = 0; a < n_act; ++a) {
        const Eigen::VectorXd v_a = V.col(a);
        const Eigen::VectorXd g_a = G.col(a);
        const double          p_a = s.p(a);
        gh.gradients.emplace_back(p_a * g_a);

        // Σ_b p_b (1 + κ_ab) w_b w_b^T = W · diag(p_b (1+κ_ab)) · W^T.
        Eigen::VectorXd weights =
            s.p.array() * (1.0 + alpha.col(a).array());
        Eigen::MatrixXd sum_w =
            W * weights.asDiagonal() * W.transpose();

        const double r_dot_ja = r_bar.dot(v_a);
        Eigen::MatrixXd H = p_a * (
            g_a * g_a.transpose()
            + (2.0 * (b_mean - beta_act(a)) + r_dot_ja)
                  * Eigen::MatrixXd::Identity(d, d)
            + r_outer
            + r_bar * v_a.transpose() + v_a * r_bar.transpose()
            - sum_w);
        // The closed form is exactly symmetric in exact arithmetic;
        // re-symmetrise to wash out floating-point asymmetry from the
        // outer products and the LDLT solve.
        H = 0.5 * (H + H.transpose());
        gh.hessians.emplace_back(std::move(H));
    }
    gh.indices = std::move(s.active);
    return gh;
}

Patch build_patch(
    int                       anchor_id,
    const Eigen::MatrixXd&    nodes,
    const std::vector<int>&   neighbor_ids,
    const Eigen::VectorXd&    beta_wpca)
{
    // ----- input validation ------------------------------------------
    if (nodes.cols() != 3) {
        throw std::invalid_argument(
            "lme::build_patch: nodes must have 3 columns (R^3 positions)");
    }
    if (anchor_id < 0 || anchor_id >= static_cast<int>(nodes.rows())) {
        throw std::invalid_argument(
            "lme::build_patch: anchor_id " + std::to_string(anchor_id)
            + " out of range [0, " + std::to_string(nodes.rows()) + ")");
    }
    if (neighbor_ids.empty()) {
        throw std::invalid_argument(
            "lme::build_patch: neighbor_ids must be non-empty");
    }
    if (beta_wpca.size() != nodes.rows()) {
        throw std::invalid_argument(
            "lme::build_patch: beta_wpca.size() must equal nodes.rows()");
    }
    for (int id : neighbor_ids) {
        if (id < 0 || id >= static_cast<int>(nodes.rows())) {
            throw std::invalid_argument(
                "lme::build_patch: neighbor id " + std::to_string(id)
                + " out of range");
        }
        if (!(beta_wpca(id) > 0.0)) {
            throw std::invalid_argument(
                "lme::build_patch: beta_wpca[" + std::to_string(id)
                + "] must be strictly positive");
        }
    }

    // ----- Gaussian weights w_b = exp(-β_b |P_b - Q|²) --------------
    // Log-sum-exp shift for numerical robustness: subtract the max log-
    // weight before exponentiating. Without it, every weight underflows
    // to zero when beta is large relative to the neighbourhood spacing.
    Patch P;
    P.anchor_id    = anchor_id;
    P.Q            = nodes.row(anchor_id).transpose();
    P.neighbor_ids = neighbor_ids;

    const Eigen::Index n_neigh =
        static_cast<Eigen::Index>(neighbor_ids.size());
    Eigen::VectorXd log_w(n_neigh);
    for (Eigen::Index k = 0; k < n_neigh; ++k) {
        const int b = neighbor_ids[static_cast<std::size_t>(k)];
        const double d2 =
            (nodes.row(b).transpose() - P.Q).squaredNorm();
        log_w(k) = -beta_wpca(b) * d2;
    }
    const double log_w_max = log_w.maxCoeff();
    Eigen::VectorXd w = (log_w.array() - log_w_max).exp();
    const double w_sum = w.sum();
    if (!(w_sum > 0.0)) {
        throw std::invalid_argument(
            "lme::build_patch: degenerate weights at anchor "
            + std::to_string(anchor_id)
            + " (sum vanished after log-sum-exp shift)");
    }
    w /= w_sum;

    // ----- weighted centroid & covariance ----------------------------
    P.Qbar = Eigen::Vector3d::Zero();
    for (Eigen::Index k = 0; k < n_neigh; ++k) {
        const int b = neighbor_ids[static_cast<std::size_t>(k)];
        P.Qbar += w(k) * nodes.row(b).transpose();
    }

    // C = Σ_k w_k (P_b - Qbar) (P_b - Qbar)^T as the symmetric outer-
    // product sum; equivalent to X_w diag(w) X_w^T where X_w columns
    // are P_b - Qbar (Millán 2011 eq. before §2.2).
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for (Eigen::Index k = 0; k < n_neigh; ++k) {
        const int b = neighbor_ids[static_cast<std::size_t>(k)];
        const Eigen::Vector3d delta =
            nodes.row(b).transpose() - P.Qbar;
        C += w(k) * (delta * delta.transpose());
    }
    C = 0.5 * (C + C.transpose());  // re-symmetrise

    // ----- eigendecomposition → tangent frame & out-of-plane eig -----
    // SelfAdjointEigenSolver returns eigenvalues in *ascending* order;
    // the top 2 (columns 1 and 2) are the in-plane axes, the bottom
    // (column 0) is the normal.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "lme::build_patch: eigendecomposition of weighted "
            "covariance failed at anchor " + std::to_string(anchor_id));
    }
    P.out_of_plane_eig = es.eigenvalues()(0);
    P.V.col(0)         = es.eigenvectors().col(1);  // larger in-plane eig
    P.V.col(1)         = es.eigenvectors().col(2);  // largest in-plane eig

    // ----- project neighbours into the tangent plane -----------------
    P.xi.resize(n_neigh, 2);
    for (Eigen::Index k = 0; k < n_neigh; ++k) {
        const int b = neighbor_ids[static_cast<std::size_t>(k)];
        const Eigen::Vector3d delta =
            nodes.row(b).transpose() - P.Qbar;
        P.xi.row(k) = (P.V.transpose() * delta).transpose();
    }
    return P;
}

ShepardWeights shepard_partition(
    const Eigen::MatrixXd&                      patch_points,
    const Eigen::VectorXd&                      beta_patches,
    const Eigen::Ref<const Eigen::Vector3d>&    x,
    double                                      tol)
{
    // ----- input validation ------------------------------------------
    if (patch_points.rows() == 0) {
        throw std::invalid_argument(
            "lme::shepard_partition: patch_points must be non-empty");
    }
    if (patch_points.cols() != 3) {
        throw std::invalid_argument(
            "lme::shepard_partition: patch_points must have 3 columns");
    }
    if (beta_patches.size() != patch_points.rows()) {
        throw std::invalid_argument(
            "lme::shepard_partition: beta_patches.size() must equal "
            "patch_points.rows()");
    }
    if (!(tol > 0.0)) {
        throw std::invalid_argument(
            "lme::shepard_partition: tol must be strictly positive (got "
            + std::to_string(tol) + ")");
    }
    for (Eigen::Index A = 0; A < beta_patches.size(); ++A) {
        if (!(beta_patches(A) > 0.0)) {
            throw std::invalid_argument(
                "lme::shepard_partition: every beta_patches[A] must be "
                "strictly positive");
        }
    }

    // ----- log-weights with LSE shift --------------------------------
    // f_A = -β_A |x - Q_A|² ; the Shepard PU then is
    //   w_A^Q(x) = exp(f_A - max f) / Σ_B exp(f_B - max f).
    const Eigen::Index M = patch_points.rows();
    Eigen::VectorXd    f(M);
    for (Eigen::Index A = 0; A < M; ++A) {
        const double d2 =
            (x.transpose() - patch_points.row(A)).squaredNorm();
        f(A) = -beta_patches(A) * d2;
    }
    const double f_max = f.maxCoeff();
    Eigen::VectorXd raw = (f.array() - f_max).exp();
    const double    sum = raw.sum();
    // sum >= 1 because at least one entry equals exp(0) = 1 after the
    // LSE shift; no zero-divide guard needed beyond what the validation
    // above already prevents.
    Eigen::VectorXd w = raw / sum;

    // ----- truncation + renormalisation ------------------------------
    ShepardWeights out;
    out.indices.reserve(static_cast<std::size_t>(M));
    out.values .reserve(static_cast<std::size_t>(M));
    double kept_sum = 0.0;
    for (Eigen::Index A = 0; A < M; ++A) {
        if (w(A) > tol) {
            out.indices.push_back(static_cast<int>(A));
            out.values .push_back(w(A));
            kept_sum += w(A);
        }
    }
    // Renormalise so the truncated support is still a partition of unity.
    // The kept_sum is strictly positive: at least the patch achieving
    // f_max has w = 1/(sum) >= 1/M > tol once tol < 1/M, and we picked
    // tol > 0 above.
    if (out.indices.empty()) {
        throw std::runtime_error(
            "lme::shepard_partition: every patch fell below tol after "
            "the log-sum-exp shift — tol is set too aggressively for "
            "this query point");
    }
    for (double& v : out.values) v /= kept_sum;
    return out;
}

CurvedBasisWeights evaluate_basis_curved(
    const std::vector<Patch>&                   patches,
    const Eigen::MatrixXd&                      patch_points,
    const Eigen::VectorXd&                      beta_patches,
    const Eigen::MatrixXd&                      nodes,
    const Eigen::VectorXd&                      beta_lme,
    const Eigen::Ref<const Eigen::Vector3d>&    y,
    double tol_shepard,
    double r_cut,
    double newton_tol,
    int    newton_max_iters)
{
    // ----- input validation ------------------------------------------
    if (patches.empty()) {
        throw std::invalid_argument(
            "lme::evaluate_basis_curved: patches must be non-empty");
    }
    if (patch_points.rows() !=
        static_cast<Eigen::Index>(patches.size())) {
        throw std::invalid_argument(
            "lme::evaluate_basis_curved: patch_points.rows() must equal "
            "patches.size()");
    }
    if (nodes.rows() == 0) {
        throw std::invalid_argument(
            "lme::evaluate_basis_curved: nodes must be non-empty");
    }
    if (beta_lme.size() != nodes.rows()) {
        throw std::invalid_argument(
            "lme::evaluate_basis_curved: beta_lme.size() must equal "
            "nodes.rows()");
    }

    // ----- Shepard PU at y -------------------------------------------
    const ShepardWeights sw =
        shepard_partition(patch_points, beta_patches, y, tol_shepard);

    // ----- accumulate per-patch contributions ------------------------
    // Compose composite weights T_a(y) = Σ_A w_A^Q(y) · p_a(Π_A(y))
    // into a flat (global_id → weight) map. The natural data structure is
    // a hash table, but the active set is small (~10-50 entries) and this
    // runs millions of times inside K assembly, so we keep two parallel
    // vectors and accumulate with a linear scan: for n ≲ 64 a branch-
    // predictable scan over contiguous ints beats an unordered_map's
    // hashing + allocation + pointer chasing (measured). The scan is O(n²)
    // in the active-set size by design — n is bounded and tiny.
    std::vector<int>    out_indices;
    std::vector<double> out_values;
    out_indices.reserve(64);
    out_values .reserve(64);

    auto find_or_insert = [&](int gid) -> std::size_t {
        // Linear scan — out_indices stays short.
        for (std::size_t k = 0; k < out_indices.size(); ++k) {
            if (out_indices[k] == gid) return k;
        }
        out_indices.push_back(gid);
        out_values .push_back(0.0);
        return out_indices.size() - 1;
    };

    for (std::size_t s = 0; s < sw.indices.size(); ++s) {
        const int     A     = sw.indices[s];
        const double  w_A   = sw.values[s];
        const Patch&  pA    = patches[static_cast<std::size_t>(A)];

        // Project y into the patch chart: ξ_y = V_A^T (y - Q̄_A).
        const Eigen::Vector3d delta = y - pA.Qbar;
        Eigen::Vector2d       xi_y  = pA.V.transpose() * delta;

        // Build the chart's β vector by slicing beta_lme on
        // pA.neighbor_ids — the 2D LME basis solve runs inside the
        // chart against those projected points.
        const Eigen::Index n_neigh =
            static_cast<Eigen::Index>(pA.neighbor_ids.size());
        Eigen::VectorXd beta_chart(n_neigh);
        for (Eigen::Index k = 0; k < n_neigh; ++k) {
            const int gid = pA.neighbor_ids[static_cast<std::size_t>(k)];
            if (gid < 0 || gid >= beta_lme.size()) {
                throw std::invalid_argument(
                    "lme::evaluate_basis_curved: neighbour id "
                    + std::to_string(gid)
                    + " out of beta_lme range");
            }
            beta_chart(k) = beta_lme(gid);
        }

        // The 2D LME basis returns p_a values in the chart-local index
        // space (0..n_neigh-1); map back to global ids via
        // pA.neighbor_ids.
        const LMEBasisValues bv = evaluate_basis(
            pA.xi, beta_chart, xi_y, r_cut,
            newton_tol, newton_max_iters);

        for (std::size_t k = 0; k < bv.indices.size(); ++k) {
            const int    local_id = bv.indices[k];
            const double p_a      = bv.values[k];
            const int    gid      = pA.neighbor_ids[
                static_cast<std::size_t>(local_id)];
            const std::size_t slot = find_or_insert(gid);
            out_values[slot] += w_A * p_a;
        }
    }

    CurvedBasisWeights out;
    out.indices = std::move(out_indices);
    out.values  = std::move(out_values);
    return out;
}

std::vector<int> prune_to_anchor_component(
    const int                            anchor,
    std::vector<int>                     nodes,
    const std::vector<std::vector<int>>& adjacency)
{
    // Membership of the selection, then BFS from the anchor walking
    // only selected nodes. The chart is small (≲ 100 nodes), so an
    // unordered_set + vector queue is plenty.
    std::unordered_set<int> selected(nodes.begin(), nodes.end());
    if (selected.count(anchor) == 0) return {anchor};

    std::unordered_set<int> reachable;
    reachable.reserve(selected.size());
    std::vector<int> queue{anchor};
    reachable.insert(anchor);
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const int u = queue[head];
        if (u < 0 || u >= static_cast<int>(adjacency.size())) continue;
        for (const int v : adjacency[static_cast<std::size_t>(u)]) {
            if (selected.count(v) != 0 && reachable.insert(v).second) {
                queue.push_back(v);
            }
        }
    }
    if (reachable.size() == selected.size()) return nodes;  // connected

    // Filter in place, preserving the input order.
    std::vector<int> kept;
    kept.reserve(reachable.size());
    for (const int v : nodes) {
        if (reachable.count(v) != 0) kept.push_back(v);
    }
    return kept;
}

namespace {

/// Shared core of the chart extractors: BFS from @p anchor expanding
/// only through nodes accepted by @p in_range (yielding the connected
/// component of the acceptance region containing the anchor), then
/// the 2-ring minimum fallback and the nearest-cap (ordered by
/// @p dist_key) + connectivity re-prune.
template <typename InRangeFn, typename DistKeyFn>
[[nodiscard]] std::vector<int> chart_bfs_core(
    const int                            anchor,
    InRangeFn&&                          in_range,
    DistKeyFn&&                          dist_key,
    const std::vector<std::vector<int>>& adjacency,
    const int                            min_nodes,
    const int                            max_nodes)
{
    const int n = static_cast<int>(adjacency.size());
    std::vector<int>        sel{anchor};
    std::unordered_set<int> visited{anchor};
    sel.reserve(64);
    for (std::size_t head = 0; head < sel.size(); ++head) {
        const int u = sel[head];
        if (u < 0 || u >= n) continue;
        for (const int v : adjacency[static_cast<std::size_t>(u)]) {
            if (visited.count(v) != 0) continue;
            if (!in_range(v)) continue;
            visited.insert(v);
            sel.push_back(v);
        }
    }

    // Minimum-chart fallback: the plain 2-ring (connected by
    // construction) so coarse/boundary anchors are never starved
    // below 2nd-order wPCA/basis support.
    if (static_cast<int>(sel.size()) < min_nodes) {
        sel.assign({anchor});
        visited.clear();
        visited.insert(anchor);
        std::vector<int> depth{0};
        for (std::size_t head = 0; head < sel.size(); ++head) {
            if (depth[head] >= 2) continue;
            const int u = sel[head];
            if (u < 0 || u >= n) continue;
            for (const int v : adjacency[static_cast<std::size_t>(u)]) {
                if (visited.insert(v).second) {
                    sel.push_back(v);
                    depth.push_back(depth[head] + 1);
                }
            }
        }
    }

    // Nearest-cap (bounds the O(n_act²) in-chart Newton; prevents the
    // high-valence freeze), then re-prune: the cap can in principle
    // disconnect the kept subset (a member's only in-set path may run
    // through a dropped farther node).
    if (static_cast<int>(sel.size()) > max_nodes) {
        std::nth_element(
            sel.begin(), sel.begin() + max_nodes, sel.end(),
            [&](int u, int w) { return dist_key(u) < dist_key(w); });
        sel.resize(static_cast<std::size_t>(max_nodes));
        sel = lme::prune_to_anchor_component(
            anchor, std::move(sel), adjacency);
    }
    return sel;
}

}  // namespace

std::vector<int> extract_chart_neighbourhood(
    const int                            anchor,
    const double                         radius,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    const int                            min_nodes,
    const int                            max_nodes)
{
    // Isotropic in-ball BFS: the connected component of (ball ∩ mesh)
    // containing the anchor — the Eq. 2 neighbourhood with the disk
    // property by construction (see the header doc). Islands and
    // antipodes are unreachable.
    const double r2 = radius * radius;
    const auto d2 = [&](int v) {
        return (nodes.row(v) - nodes.row(anchor)).squaredNorm();
    };
    return chart_bfs_core(
        anchor,
        [&](int v) { return d2(v) <= r2; },
        d2,
        adjacency, min_nodes, max_nodes);
}

std::vector<int> extract_chart_neighbourhood_directional(
    const int                            anchor,
    const double                         mult,
    const std::vector<Eigen::Vector3d>&  anchor_edges,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    const int                            min_nodes,
    const int                            max_nodes)
{
    // Anisotropic variant: in-range iff the offset, measured in units
    // of the one-ring spacing IN ITS OWN DIRECTION (the faithful
    // max-projection spacing h_a(u), RMA13 §3.2), is at most mult:
    //   ||y|| <= mult·h_a(ŷ)  ⇔  ||y||² <= mult·max_b|e_b·y|.
    // See the header doc for the rationale (k-ring-equivalent reach
    // on anisotropic grids; flat-sheet folds metrically excluded).
    const Eigen::Vector3d xa = nodes.row(anchor).transpose();
    const auto h_dir_scaled = [&](const Eigen::Vector3d& y) {
        double m = 0.0;
        for (const Eigen::Vector3d& e : anchor_edges)
            m = std::max(m, std::abs(e.dot(y)));
        return m;  // = h_a(ŷ)·||y||
    };
    return chart_bfs_core(
        anchor,
        [&](int v) {
            const Eigen::Vector3d y = nodes.row(v).transpose() - xa;
            return y.squaredNorm() <= mult * h_dir_scaled(y);
        },
        [&](int v) {
            const Eigen::Vector3d y = nodes.row(v).transpose() - xa;
            const double m = h_dir_scaled(y);
            return m > 0.0 ? y.squaredNorm() / m
                           : std::numeric_limits<double>::infinity();
        },
        adjacency, min_nodes, max_nodes);
}

std::vector<int> extract_chart_neighbourhood_intrinsic(
    const int                            anchor,
    const double                         radius,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    const int                            min_nodes,
    const int                            max_nodes)
{
    const int n = static_cast<int>(adjacency.size());
    // Dijkstra over edge lengths, truncated at the geodesic radius.
    // Charts are small (≲ 100 nodes), so a binary-heap queue with a
    // settled-set is plenty; stale queue entries are skipped.
    std::unordered_map<int, double> dist;
    std::unordered_set<int>         settled;
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>,
                        std::greater<>> queue;
    dist.emplace(anchor, 0.0);
    queue.emplace(0.0, anchor);
    std::vector<int> sel;
    sel.reserve(64);
    while (!queue.empty()) {
        const auto [d, u] = queue.top();
        queue.pop();
        if (!settled.insert(u).second) continue;
        sel.push_back(u);
        if (u < 0 || u >= n) continue;
        for (const int v : adjacency[static_cast<std::size_t>(u)]) {
            if (settled.count(v) != 0) continue;
            const double dv =
                d + (nodes.row(v) - nodes.row(u)).norm();
            if (dv > radius) continue;
            const auto it = dist.find(v);
            if (it == dist.end()) {
                dist.emplace(v, dv);
                queue.emplace(dv, v);
            } else if (dv < it->second) {
                it->second = dv;
                queue.emplace(dv, v);
            }
        }
    }

    // 2-ring minimum fallback / intrinsic nearest-cap, mirroring the
    // extrinsic extractors. The capped set is a smaller Dijkstra ball
    // (every shortest-path prefix has smaller distance), hence still
    // connected; the re-prune stays as a cheap unconditional
    // guarantee. Both cases delegate to chart_bfs_core over the
    // already-computed distance field.
    if (static_cast<int>(sel.size()) < min_nodes ||
        static_cast<int>(sel.size()) > max_nodes) {
        return chart_bfs_core(
            anchor,
            [&](int v) { return dist.count(v) != 0; },
            [&](int v) {
                const auto it = dist.find(v);
                return it != dist.end()
                           ? it->second
                           : std::numeric_limits<double>::infinity();
            },
            adjacency, min_nodes, max_nodes);
    }
    return sel;
}

}  // namespace lme

namespace {

/// Shepard partition-of-unity truncation tolerance (Millán 2011 Table I,
/// their Eq. 3 "0 otherwise"). Shared by the stiffness AND mass assembly so
/// K and M always live on the same function space — a mismatch produces
/// spurious modes (see the ShellAssembler contract). Do NOT fork this per
/// path: the earlier ghost-on M override to 1e-3 silently broke that
/// invariant. The out-of-hull failure that override patched is now
/// structurally prevented by the per-patch quadrature ownership filter.
constexpr double kPoUTruncTol = 1.0e-6;

/// Per-vertex local spacing h_a = mean length of edges incident to
/// vertex a. Drives the per-node locality parameter β_a = γ / h_a²
/// (Millan 2011 §2.4). Returned vector has length @c V.rows().
[[nodiscard]] Eigen::VectorXd vertex_one_ring_h(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    const Eigen::Index n_v = V.rows();
    Eigen::VectorXd  sum(n_v);  sum.setZero();
    Eigen::VectorXi  cnt(n_v);  cnt.setZero();

    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int i = F(t, k);
            const int j = F(t, (k + 1) % 3);
            const double L = (V.row(i) - V.row(j)).norm();
            sum(i) += L;  cnt(i) += 1;
            sum(j) += L;  cnt(j) += 1;
        }
    }

    Eigen::VectorXd h(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        if (cnt(a) == 0) {
            throw std::invalid_argument(
                "LMEAssembler: vertex " + std::to_string(a)
                + " is not referenced by any triangle in F");
        }
        h(a) = sum(a) / static_cast<double>(cnt(a));
        // A zero mean edge length (every incident edge degenerate, i.e.
        // coincident vertices) would flow into inv_h = 1/h_chart and
        // inject Inf into the basis. Localise the error here instead of
        // leaking a non-finite into K/M.
        if (!(h(a) > 0.0)) {
            throw std::invalid_argument(
                "LMEAssembler: vertex " + std::to_string(a)
                + " has zero mean edge length (coincident vertices?)");
        }
    }
    return h;
}

/// Per-vertex one-ring edge vectors @f$ \{x_b - x_a : b \in N(a)\} @f$ in
/// world coordinates. The faithful nodal-spacing metric (RMA13 §3.2.1–§3.2.2)
/// needs the *directional* spacing @f$ h_a(u) = \max_b |(x_b-x_a)\cdot u| @f$
/// along a query direction @c u — the representative MAX-adjacent spacing the
/// paper's 1D rule prescribes — which the covariance (a *mean* of squared edge
/// projections) cannot recover (it under-measures by a valence factor and
/// leaks perpendicular edges into each axis). The per-chart gap builder
/// projects these edges into each chart frame and takes the max projection.
[[nodiscard]] std::vector<std::vector<Eigen::Vector3d>> vertex_one_ring_edges(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    const Eigen::Index n_v = V.rows();
    std::vector<std::vector<Eigen::Vector3d>> E(
        static_cast<std::size_t>(n_v));
    // Dedup neighbours via a per-vertex set so a shared edge is recorded once.
    std::vector<std::unordered_set<int>> seen(static_cast<std::size_t>(n_v));
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int i = F(t, k);
            const int j = F(t, (k + 1) % 3);
            if (seen[static_cast<std::size_t>(i)].insert(j).second) {
                E[static_cast<std::size_t>(i)].push_back(
                    (V.row(j) - V.row(i)).transpose());
            }
            if (seen[static_cast<std::size_t>(j)].insert(i).second) {
                E[static_cast<std::size_t>(j)].push_back(
                    (V.row(i) - V.row(j)).transpose());
            }
        }
    }
    return E;
}

/// Per-vertex one-ring spacing covariance @f$ C_a = \langle (x_b - x_a)
/// \otimes (x_b - x_a) \rangle_{b \in N(a)} @f$ — the (unnormalised)
/// nodal-spacing metric tensor of Rosolen–Millán–Arroyo 2013 §3.2.2.
/// Its eigenvectors give the principal spacing directions and its
/// eigenvalue ratio the local mesh anisotropy; the magnitude is
/// normalised away at the gap-construction site (the absolute scale is
/// carried by @ref vertex_one_ring_h, matching the isotropic recipe).
/// 3x3 in world coordinates; the curved-shell gap builder projects it
/// into each chart frame. Ghost nodes have no one-ring and are handled
/// isotropically by the caller, so they are not represented here.
[[nodiscard]] std::vector<Eigen::Matrix3d> vertex_one_ring_covariance(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    const Eigen::Index n_v = V.rows();
    std::vector<Eigen::Matrix3d> C(
        static_cast<std::size_t>(n_v), Eigen::Matrix3d::Zero());
    Eigen::VectorXi cnt(n_v);  cnt.setZero();

    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int i = F(t, k);
            const int j = F(t, (k + 1) % 3);
            const Eigen::Vector3d e = (V.row(j) - V.row(i)).transpose();
            const Eigen::Matrix3d ee = e * e.transpose();
            C[static_cast<std::size_t>(i)] += ee;  cnt(i) += 1;
            C[static_cast<std::size_t>(j)] += ee;  cnt(j) += 1;
        }
    }
    for (Eigen::Index a = 0; a < n_v; ++a) {
        if (cnt(a) > 0) {
            C[static_cast<std::size_t>(a)] /=
                static_cast<double>(cnt(a));
        }
    }
    return C;
}

/// Check that the mesh is essentially a flat 2D plate in the xy-plane.
/// Guards the legacy flat-plate assembly path only. The curved-shell path
/// (wPCA + Shepard PU, Millan 2011 §3) is the shipped default — see
/// @ref LMEAssembler::Params::use_curved_shell — and does not call this.
/// Throws @c std::invalid_argument if z varies appreciably.
void require_flat_plate(const Eigen::MatrixXd& V)
{
    if (V.cols() < 2) {
        throw std::invalid_argument(
            "LMEAssembler: V must have at least 2 columns (xy positions)");
    }
    if (V.cols() == 2) {
        return;  // pure 2D — nothing to check
    }
    const double zmin = V.col(2).minCoeff();
    const double zmax = V.col(2).maxCoeff();
    const double xy_extent =
        (V.col(0).maxCoeff() - V.col(0).minCoeff())
        + (V.col(1).maxCoeff() - V.col(1).minCoeff());
    const double tol = 1e-9 * std::max(1.0, xy_extent);
    if (zmax - zmin > tol) {
        throw std::invalid_argument(
            "LMEAssembler: currently only flat-plate meshes are supported "
            "(z-coordinates must be constant). Curved-shell support via "
            "wPCA + Shepard PU is a planned follow-up (Millan 2011 §3).");
    }
}

/// Twice the signed area of a 2D triangle (V0, V1, V2) — i.e. the
/// determinant of [V1 - V0; V2 - V0]. Returned as a signed double; the
/// caller takes the absolute value as needed.
[[nodiscard]] double signed_double_area_2d(
    const Eigen::RowVector2d& V0,
    const Eigen::RowVector2d& V1,
    const Eigen::RowVector2d& V2) noexcept
{
    const Eigen::RowVector2d e01 = V1 - V0;
    const Eigen::RowVector2d e02 = V2 - V0;
    return e01.x() * e02.y() - e01.y() * e02.x();
}

/// Vertex-vertex adjacency from the triangle table. Each entry is
/// deduplicated (a single edge contributes at most once to each
/// endpoint's neighbour list). Cost @f$O(F + V)@f$ amortised.
[[nodiscard]] std::vector<std::vector<int>> build_vertex_adjacency(
    Eigen::Index n_v,
    const Eigen::MatrixXi& F)
{
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n_v));
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int i = F(t, k);
            const int j = F(t, (k + 1) % 3);
            adj[static_cast<std::size_t>(i)].push_back(j);
            adj[static_cast<std::size_t>(j)].push_back(i);
        }
    }
    for (auto& nbrs : adj) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return adj;
}

/// BFS the @c k_max ring around @p anchor through the vertex
/// adjacency. Returns the visited vertices in any order — including
/// @p anchor itself. Used to localise per-patch neighbourhoods to a
/// topology-bounded region of the mesh so that far-away (e.g.
/// antipodal-on-sphere) vertices never project onto the chart and
/// degenerate the in-chart LME basis solve.
[[nodiscard]] std::vector<int> k_ring(
    int                                  anchor,
    int                                  k_max,
    const std::vector<std::vector<int>>& adjacency)
{
    const int n_v = static_cast<int>(adjacency.size());
    std::vector<int> depth(static_cast<std::size_t>(n_v), -1);
    depth[static_cast<std::size_t>(anchor)] = 0;
    std::queue<int>  q;
    q.push(anchor);
    while (!q.empty()) {
        const int u = q.front();  q.pop();
        const int d = depth[static_cast<std::size_t>(u)];
        if (d >= k_max) continue;
        for (int v : adjacency[static_cast<std::size_t>(u)]) {
            if (depth[static_cast<std::size_t>(v)] == -1) {
                depth[static_cast<std::size_t>(v)] = d + 1;
                q.push(v);
            }
        }
    }
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n_v));
    for (int v = 0; v < n_v; ++v) {
        if (depth[static_cast<std::size_t>(v)] != -1) out.push_back(v);
    }
    return out;
}

/// Per-vertex GLOBAL-boundary frame for the faithful SME §3.2.2 nodal-
/// gap recipe (Rosolen–Millán–Arroyo 2013, Fig. 6). Classifies each real
/// vertex relative to the mesh boundary and carries the world-space
/// boundary tangent/normal that the per-chart gap builder projects into
/// each chart frame. Replaces the previous "every boundary-touching
/// vertex → d=0" rule, which over-shrinks the moment hull on open meshes
/// and makes near-free-edge charts 2nd-order INFEASIBLE.
struct VtxBoundaryFrame {
    lme::NodalGapKind kind = lme::NodalGapKind::Interior;
    Eigen::Vector3d   t  = Eigen::Vector3d::Zero();   ///< boundary tangent
    Eigen::Vector3d   n  = Eigen::Vector3d::Zero();   ///< in-surface bdry normal
    Eigen::Vector3d   n2 = Eigen::Vector3d::Zero();   ///< 2nd normal (corner)
};

/// Classify every vertex per paper §3.2.2 Fig. 6:
///  - on a boundary edge, between two near-collinear boundary edges
///    → BoundaryEdgeMid (gap (α/4)h² t⊗t, no normal slack — keeping the
///    weak-Kronecker-delta property at the rim, paper Eq. for AB);
///  - where two boundary edges meet at a sharp angle → BoundaryCorner
///    (d=0, exact interpolation at domain corners);
///  - interior but adjacent to the boundary → NearOneBoundaryEdge
///    (β h² n⊗n + (α/4)h² t⊗t) or, next to two boundary faces,
///    NearTwoBoundaryEdges (β h²(n⊗n + n'⊗n'));
///  - otherwise Interior.
[[nodiscard]] std::vector<VtxBoundaryFrame> compute_sme_boundary_frames(
    const Eigen::MatrixXd&                        V,
    const std::vector<lme::BoundaryEdge>&         bdry,
    const std::vector<std::vector<int>>&          adjacency,
    Eigen::Index                                  n_v)
{
    constexpr double kCornerCos = 0.70710678;  // 45° turn ⇒ corner
    std::vector<VtxBoundaryFrame> fr(static_cast<std::size_t>(n_v));
    std::vector<char> is_bdry(static_cast<std::size_t>(n_v), 0);

    // Tangent (edge direction) and in-surface inward normal of one
    // boundary edge.
    auto edge_tn = [&](const lme::BoundaryEdge& be,
                       Eigen::Vector3d& t, Eigen::Vector3d& n) {
        const Eigen::Vector3d e0 = V.row(be.v0).transpose();
        const Eigen::Vector3d e1 = V.row(be.v1).transpose();
        t = (e1 - e0).normalized();
        n = Eigen::Vector3d::Zero();
        if (be.v_int >= 0) {
            const Eigen::Vector3d vi = V.row(be.v_int).transpose();
            const Eigen::Vector3d into = vi - 0.5 * (e0 + e1);
            const Eigen::Vector3d np = into - into.dot(t) * t;
            const double nn = np.norm();
            if (nn > 1.0e-12) n = np / nn;
        }
    };

    using TN = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
    std::vector<std::vector<TN>> btn(static_cast<std::size_t>(n_v));
    for (const auto& be : bdry) {
        is_bdry[static_cast<std::size_t>(be.v0)] = 1;
        is_bdry[static_cast<std::size_t>(be.v1)] = 1;
        Eigen::Vector3d t, n; edge_tn(be, t, n);
        btn[static_cast<std::size_t>(be.v0)].push_back({t, n});
        btn[static_cast<std::size_t>(be.v1)].push_back({t, n});
    }

    // Boundary vertices: corner vs mid-edge from incident-edge angle.
    for (Eigen::Index v = 0; v < n_v; ++v) {
        auto& inc = btn[static_cast<std::size_t>(v)];
        if (inc.empty()) continue;
        const Eigen::Vector3d tref = inc[0].first;
        Eigen::Vector3d tsum = Eigen::Vector3d::Zero();
        bool corner = false;
        for (auto& tn : inc) {
            const double s = tn.first.dot(tref) < 0 ? -1.0 : 1.0;
            tsum += s * tn.first;
            if (std::abs(tn.first.dot(tref)) < kCornerCos) corner = true;
        }
        if (corner) {
            fr[static_cast<std::size_t>(v)].kind =
                lme::NodalGapKind::BoundaryCorner;
        } else if (tsum.norm() > 1.0e-12) {
            fr[static_cast<std::size_t>(v)].kind =
                lme::NodalGapKind::BoundaryEdgeMid;
            fr[static_cast<std::size_t>(v)].t = tsum.normalized();
        } else {
            fr[static_cast<std::size_t>(v)].kind =
                lme::NodalGapKind::BoundaryCorner;
        }
    }

    // Boundary-edge vertices FLANKING a corner get the larger tangential
    // slack β h² t⊗t (paper §3.2.2: "d_a = β h²_a t⊗t for the nodes next
    // to A and B"), vs the (α/4)h² t⊗t of mid-edge nodes. With β ≥ 1 >
    // α/4 (at the paper's α=2) this extra tangential slack near the corner
    // is what keeps the corner region 2nd-order FEASIBLE — the case whose
    // omission previously forced an unfaithful LME fallback there.
    {
        std::vector<std::vector<int>> bnbr(static_cast<std::size_t>(n_v));
        for (const auto& be : bdry) {
            bnbr[static_cast<std::size_t>(be.v0)].push_back(be.v1);
            bnbr[static_cast<std::size_t>(be.v1)].push_back(be.v0);
        }
        for (Eigen::Index v = 0; v < n_v; ++v) {
            if (fr[static_cast<std::size_t>(v)].kind
                != lme::NodalGapKind::BoundaryEdgeMid) continue;
            for (int u : bnbr[static_cast<std::size_t>(v)]) {
                if (fr[static_cast<std::size_t>(u)].kind
                        == lme::NodalGapKind::BoundaryCorner) {
                    fr[static_cast<std::size_t>(v)].kind =
                        lme::NodalGapKind::BoundaryEdgeNearCorner;
                    break;
                }
            }
        }
    }

    // Interior vertices adjacent to the boundary: gather boundary (t,n)
    // from boundary neighbours; one vs two distinct boundary normals.
    for (Eigen::Index v = 0; v < n_v; ++v) {
        if (is_bdry[static_cast<std::size_t>(v)]) continue;
        std::vector<TN> assoc;
        for (int u : adjacency[static_cast<std::size_t>(v)]) {
            if (u >= 0 && u < n_v
                && is_bdry[static_cast<std::size_t>(u)])
                for (auto& tn : btn[static_cast<std::size_t>(u)])
                    assoc.push_back(tn);
        }
        if (assoc.empty()) continue;
        Eigen::Vector3d nref = Eigen::Vector3d::Zero();
        for (auto& tn : assoc) if (tn.second.norm() > 1e-9) { nref = tn.second; break; }
        if (nref.norm() < 1e-9) continue;
        const Eigen::Vector3d tref = assoc[0].first;
        Eigen::Vector3d nsum = Eigen::Vector3d::Zero();
        Eigen::Vector3d tsum = Eigen::Vector3d::Zero();
        Eigen::Vector3d n_other = Eigen::Vector3d::Zero();
        bool two = false;
        for (auto& tn : assoc) {
            if (tn.second.norm() < 1e-9) continue;
            nsum += (tn.second.dot(nref) < 0 ? -1.0 : 1.0) * tn.second;
            tsum += (tn.first.dot(tref) < 0 ? -1.0 : 1.0) * tn.first;
            if (std::abs(tn.second.dot(nref)) < kCornerCos) {
                two = true; n_other = tn.second;
            }
        }
        auto& f = fr[static_cast<std::size_t>(v)];
        if (two && n_other.norm() > 1e-9) {
            f.kind = lme::NodalGapKind::NearTwoBoundaryEdges;
            f.n  = nref.normalized();
            f.n2 = n_other.normalized();
        } else if (nsum.norm() > 1e-9) {
            f.kind = lme::NodalGapKind::NearOneBoundaryEdge;
            f.n = nsum.normalized();
            f.t = tsum.norm() > 1e-9 ? tsum.normalized()
                                     : Eigen::Vector3d::Zero();
        }
    }
    return fr;
}

/// Build the per-patch 2nd-order SME nodal-gap matrices @f$ d_a @f$
/// (Rosolen–Millán–Arroyo 2013 §3.2.2, Fig. 6) for every patch in
/// @p patches, in chart-2D coordinates.
///
/// Classification per node comes from its GLOBAL-boundary frame
/// (@p frames, see @ref compute_sme_boundary_frames) — NOT from the
/// chart's k-ring rim:
///  - Ghost nodes (global id @c >= n_v): @ref lme::NodalGapKind::Interior
///    (they extend the cloud past the rim and carry no boundary frame).
///  - Interior: anisotropy-oriented @f$(\alpha/4) h^2 I@f$, with the
///    gap rotated to the chart-projected one-ring spacing metric.
///  - Boundary edge: @f$(\alpha/4) h^2\, t\otimes t@f$ (tangential slack
///    only); flanking a corner: @f$\beta h^2\, t\otimes t@f$.
///  - Boundary corner: @f$ d_a = 0 @f$ (exact interpolation).
///  - Next to one boundary edge: @f$\beta h^2\, n\otimes n +
///    (\alpha/4) h^2\, t\otimes t@f$; next to two:
///    @f$\beta h^2 (n\otimes n + n'\otimes n')@f$.
/// The world boundary tangent/normal are projected into each chart's
/// wPCA frame and renormalised here.
///
/// Each gap direction is scaled by the TRUE DIRECTIONAL nodal spacing
/// @f$ h_a(u) = \max_b |(x_b - x_a)\cdot u| @f$ — the paper's max-adjacent
/// rule (RMA13 §3.2.1) generalised to a query direction — computed from the
/// chart-projected one-ring (@p one_ring_edges): interior eigen-directions
/// (@f$ h^i @f$ → @ref lme::NodalGapKind::InteriorAnisotropic) and the
/// boundary normal/tangent (@ref lme::NodalGap::h_n, @ref lme::NodalGap::h_t)
/// alike. This replaces the earlier trace-normalised-covariance × mean-edge
/// scalar, which discarded the per-direction scale and under-covered the
/// sparse axis on graded meshes (the free-free-cylinder feasibility failure).
/// The isotropic scalar @p h_a_ext (3D one-ring mean) is retained as the
/// fallback when the spacing is isotropic or a one-ring is unavailable (e.g.
/// ghost nodes), keeping quasi-uniform meshes byte-identical. On a flat plate
/// the chart projection is an isometry so the spacing is exact; on a curved
/// chart it is correct up to a second-order curvature term.
[[nodiscard]] std::vector<std::vector<Eigen::Matrix2d>>
build_per_patch_sme_gaps(
    const std::vector<lme::Patch>&                   patches,
    const std::vector<VtxBoundaryFrame>&             frames,
    Eigen::Index                                     n_v,
    const Eigen::VectorXd&                           h_a_ext,
    const std::vector<Eigen::Matrix3d>&              node_cov,
    const std::vector<std::vector<Eigen::Vector3d>>& one_ring_edges,
    double                                           sme_alpha,
    double                                           sme_beta)
{
    // FAITHFUL paper §3.2.2 (Fig. 6) gap classification. A vertex's gap
    // kind comes from its GLOBAL mesh-boundary frame (precomputed in
    // @ref compute_sme_boundary_frames), NOT from the chart's k-ring rim:
    //  - interior            → (α/4) h² I (anisotropy-oriented below);
    //  - boundary edge       → (α/4) h² t⊗t   (tangential slack only);
    //  - boundary corner     → 0              (exact interpolation);
    //  - next to boundary    → β h² n⊗n + (α/4) h² t⊗t;
    //  - next to two faces   → β h² (n⊗n + n'⊗n').
    // The boundary tangent/normal live in the surface; they are projected
    // into each chart's wPCA frame and renormalised here. The earlier
    // "every boundary-touching vertex → d=0" rule (plus a chart-outer-ring
    // d=0 hack) over-shrank the moment hull and made near-free-edge charts
    // 2nd-order INFEASIBLE; the graded recipe restores feasibility while
    // keeping the weak-Kronecker-delta property at true corners/edges.
    constexpr double kAnisoIsotropicTol = 1.0 + 1.0e-6;
    // A surface tangent/normal projects to < this fraction of unit length
    // in a chart only when it is nearly perpendicular to the chart plane
    // (steeply-curved chart). Treat as degenerate and fall back.
    constexpr double kChartProjMin = 0.25;

    // ----- LME_DIAG classification accounting (env-gated) ------------
    // A node's gap KIND is global (its mesh-boundary frame), but the gap
    // MATRIX is chart-local: the frame is projected into each chart's
    // wPCA plane with hard fallbacks when the projection degenerates
    // (kChartProjMin) and a binary aniso/iso gate (kAnisoIsotropicTol).
    // These tallies make the chart-shape SENSITIVITY of that projection
    // layer measurable — per-(patch,node) final-kind histogram, fallback
    // counts, accepted-projection extremes, and the number of vertices
    // whose EFFECTIVE kind differs across the charts containing them
    // (the suspected discontinuity behind the non-k-ring chart A/B
    // erraticism, inventory C4 round 2/3).
    const bool diag_cls = std::getenv("LME_DIAG") != nullptr;
    struct ClsStats {
        long kind[7]       = {0, 0, 0, 0, 0, 0, 0};
        long fb_t_corner   = 0;  ///< EdgeMid/EdgeNC: t-proj fail → Corner
        long fb_n_interior = 0;  ///< Near1: n-proj fail → interior
        long fb_two_one    = 0;  ///< Near2: one normal fail → Near1
        long fb_two_int    = 0;  ///< Near2: both normals fail → interior
        long gate_aniso    = 0;  ///< fill_interior: anisotropic branch
        long gate_iso      = 0;  ///< fill_interior: isotropic branch
        double min_proj    = 1.0;  ///< smallest ACCEPTED projection norm
        long proj_near     = 0;  ///< accepted projections in [0.25, 0.40)
        double hn_h_max    = 0.0, ht_h_max = 0.0;  ///< max h_n/h, h_t/h
    } cls;
    std::vector<std::uint8_t> cls_kind_mask;  // per-gid OR of 1<<kind
    std::vector<int>          cls_n_seen;     // per-gid #charts containing it
    if (diag_cls) {
        cls_kind_mask.assign(static_cast<std::size_t>(n_v), 0);
        cls_n_seen.assign(static_cast<std::size_t>(n_v), 0);
    }

    std::vector<std::vector<Eigen::Matrix2d>> out;
    out.reserve(patches.size());
    for (const auto& pA : patches) {
        const Eigen::Index n_act =
            static_cast<Eigen::Index>(pA.neighbor_ids.size());

        // Project a world surface direction into this chart frame and
        // renormalise; false if it is too close to the chart normal.
        auto to_chart = [&](const Eigen::Vector3d& w,
                            Eigen::Vector2d& out2) -> bool {
            out2 = pA.V.transpose() * w;
            const double nn = out2.norm();
            if (nn < kChartProjMin) return false;
            if (diag_cls) {
                cls.min_proj = std::min(cls.min_proj, nn);
                if (nn < 0.40) ++cls.proj_near;
            }
            out2 /= nn;
            return true;
        };
        // Directional nodal spacing h_a(u) = max_b |(x_b - x_a)·u| over node
        // gid's one-ring, in this chart's 2D coords (RMA13 §3.2.1's MAX-
        // adjacent rule, generalised to a query direction u). Returns 0 when
        // the node has no recorded one-ring (e.g. a ghost), so callers fall
        // back to the isotropic h.
        auto dir_spacing = [&](int gid, const Eigen::Vector2d& u) -> double {
            if (gid < 0 || gid >= static_cast<int>(one_ring_edges.size()))
                return 0.0;
            double hmax = 0.0;
            for (const Eigen::Vector3d& e :
                 one_ring_edges[static_cast<std::size_t>(gid)]) {
                const Eigen::Vector2d e2 = pA.V.transpose() * e;
                hmax = std::max(hmax, std::abs(e2.dot(u)));
            }
            return hmax;
        };
        // Anisotropy-oriented interior gap for node gid (RMA13 §3.2.2
        // anisotropic case): d_a = (α/4) Σ_i (h^i)² v^i⊗v^i with v^i the
        // chart-projected one-ring principal spacing directions (covariance
        // eigenvectors) and (h^i) the TRUE directional spacing along each —
        // the max-adjacent projection, NOT the trace-normalised covariance
        // magnitude (which discards the per-direction scale). On an
        // elongated cloud this gives the sparse axis its full slack. Falls
        // back to the isotropic (α/4)h²I when the spacing is isotropic or
        // the one-ring is unavailable (keeps quasi-uniform meshes untouched).
        auto fill_interior = [&](lme::NodalGap& g, int gid) {
            const Eigen::Matrix2d C_chart =
                pA.V.transpose()
                * node_cov[static_cast<std::size_t>(gid)]
                * pA.V;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(C_chart);
            const Eigen::Vector2d lam = es.eigenvalues();   // ascending
            const double lam_lo = lam(0), lam_hi = lam(1);
            const Eigen::Vector2d v1 = es.eigenvectors().col(0);
            const Eigen::Vector2d v2 = es.eigenvectors().col(1);
            const double h1 = dir_spacing(gid, v1);
            const double h2 = dir_spacing(gid, v2);
            if (lam_lo > 0.0 && lam_hi / lam_lo > kAnisoIsotropicTol
                && h1 > 0.0 && h2 > 0.0) {
                g.kind = lme::NodalGapKind::InteriorAnisotropic;
                g.h1   = h1;
                g.h2   = h2;
                g.v1   = v1;
                g.v2   = v2;
                if (diag_cls) ++cls.gate_aniso;
            } else {
                g.kind = lme::NodalGapKind::Interior;
                if (diag_cls) ++cls.gate_iso;
            }
        };

        std::vector<lme::NodalGap> info;
        info.reserve(static_cast<std::size_t>(n_act));
        for (Eigen::Index k = 0; k < n_act; ++k) {
            const int gid = pA.neighbor_ids[
                static_cast<std::size_t>(k)];
            lme::NodalGap g{};
            g.h = h_a_ext(gid);
            const bool is_ghost = (gid >= static_cast<int>(n_v));
            if (is_ghost) {
                // Ghost nodes extend the cloud past the rim and carry no
                // boundary frame; keep the isotropic interior gap.
                g.kind = lme::NodalGapKind::Interior;
                info.push_back(g);
                continue;
            }
            const VtxBoundaryFrame& fr =
                frames[static_cast<std::size_t>(gid)];
            Eigen::Vector2d t2, n2, n2b;
            switch (fr.kind) {
            case lme::NodalGapKind::BoundaryCorner:
                g.kind = lme::NodalGapKind::BoundaryCorner;
                break;
            case lme::NodalGapKind::BoundaryEdgeMid:
                if (to_chart(fr.t, t2)) {
                    g.kind = lme::NodalGapKind::BoundaryEdgeMid;
                    g.t = t2;
                    g.h_t = dir_spacing(gid, t2);
                } else {
                    g.kind = lme::NodalGapKind::BoundaryCorner;
                    if (diag_cls) ++cls.fb_t_corner;
                }
                break;
            case lme::NodalGapKind::BoundaryEdgeNearCorner:
                if (to_chart(fr.t, t2)) {
                    g.kind = lme::NodalGapKind::BoundaryEdgeNearCorner;
                    g.t = t2;
                    g.h_t = dir_spacing(gid, t2);
                } else {
                    g.kind = lme::NodalGapKind::BoundaryCorner;
                    if (diag_cls) ++cls.fb_t_corner;
                }
                break;
            case lme::NodalGapKind::NearOneBoundaryEdge: {
                const bool ok_n = to_chart(fr.n, n2);
                if (ok_n) {
                    Eigen::Vector2d tt;
                    const bool ok_t = (fr.t.norm() > 1e-9)
                                      && to_chart(fr.t, tt);
                    g.kind = lme::NodalGapKind::NearOneBoundaryEdge;
                    g.n = n2;
                    // Tangent ⟂ normal in-chart; synthesise if the
                    // projected tangent is unavailable/degenerate.
                    g.t = ok_t ? tt
                               : Eigen::Vector2d(-n2.y(), n2.x());
                    // Directional spacing along the (sparse) normal and the
                    // tangent — the β h_n² normal slack must cover the larger
                    // axial spacing on an anisotropic free-edge mesh.
                    g.h_n = dir_spacing(gid, g.n);
                    g.h_t = dir_spacing(gid, g.t);
                } else {
                    fill_interior(g, gid);
                    if (diag_cls) ++cls.fb_n_interior;
                }
                break;
            }
            case lme::NodalGapKind::NearTwoBoundaryEdges: {
                const bool ok_a = to_chart(fr.n,  n2);
                const bool ok_b = to_chart(fr.n2, n2b);
                if (ok_a && ok_b) {
                    g.kind = lme::NodalGapKind::NearTwoBoundaryEdges;
                    g.n = n2; g.n2 = n2b;
                    // Both normals share the β h_n² term; take the larger
                    // directional spacing (conservative enlargement).
                    g.h_n = std::max(dir_spacing(gid, n2),
                                     dir_spacing(gid, n2b));
                } else if (ok_a) {
                    g.kind = lme::NodalGapKind::NearOneBoundaryEdge;
                    g.n = n2;
                    g.t = Eigen::Vector2d(-n2.y(), n2.x());
                    g.h_n = dir_spacing(gid, g.n);
                    g.h_t = dir_spacing(gid, g.t);
                    if (diag_cls) ++cls.fb_two_one;
                } else {
                    fill_interior(g, gid);
                    if (diag_cls) ++cls.fb_two_int;
                }
                break;
            }
            default:
                fill_interior(g, gid);
                break;
            }
            if (diag_cls) {
                ++cls.kind[static_cast<std::size_t>(g.kind)];
                cls_kind_mask[static_cast<std::size_t>(gid)] |=
                    static_cast<std::uint8_t>(
                        1u << static_cast<unsigned>(g.kind));
                ++cls_n_seen[static_cast<std::size_t>(gid)];
                if (g.h > 0.0) {
                    if (g.h_n > 0.0)
                        cls.hn_h_max = std::max(cls.hn_h_max, g.h_n / g.h);
                    if (g.h_t > 0.0)
                        cls.ht_h_max = std::max(cls.ht_h_max, g.h_t / g.h);
                }
            }
            info.push_back(g);
        }
        out.push_back(
            lme::compute_nodal_gaps(info, sme_alpha, sme_beta));
    }
    if (diag_cls) {
        // A vertex whose final kind differs between two charts that both
        // contain it = a chart-shape-induced classification flip (the
        // gap matrix d_a it receives is discontinuous in chart
        // membership there). mask & (mask-1) != 0 ⇔ >1 distinct kind.
        long n_multi = 0, n_flip = 0;
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const auto m = cls_kind_mask[static_cast<std::size_t>(v)];
            if (cls_n_seen[static_cast<std::size_t>(v)] > 1 && m != 0) {
                ++n_multi;
                if ((m & static_cast<std::uint8_t>(m - 1)) != 0) ++n_flip;
            }
        }
        std::fprintf(stderr,
            "[lme-diag][sme-cls] kinds(per patch-node, real verts): Int=%ld "
            "IntAniso=%ld Corner=%ld EdgeMid=%ld EdgeNC=%ld Near1=%ld "
            "Near2=%ld\n"
            "[lme-diag][sme-cls] fallbacks: t->Corner=%ld "
            "Near1->interior=%ld Near2->Near1=%ld Near2->interior=%ld | "
            "accepted proj: min=%.3f n[0.25,0.40)=%ld\n"
            "[lme-diag][sme-cls] interior gate: aniso=%ld iso=%ld | "
            "h_n/h max=%.2f h_t/h max=%.2f | kind FLIPS across charts: "
            "%ld of %ld multi-chart verts\n",
            cls.kind[0], cls.kind[1], cls.kind[2], cls.kind[3],
            cls.kind[4], cls.kind[5], cls.kind[6],
            cls.fb_t_corner, cls.fb_n_interior, cls.fb_two_one,
            cls.fb_two_int, cls.min_proj, cls.proj_near,
            cls.gate_aniso, cls.gate_iso, cls.hn_h_max, cls.ht_h_max,
            n_flip, n_multi);
    }
    return out;
}

/// Map @ref LMEAssembler::Params::n_quadrature_per_tri (a sample
/// count) onto the @ref QuadratureRule enum used by
/// @ref chladni::shell::quadrature_points. The paper's rule is the
/// 12-point degree-6 Dunavant (Millán 2011 §4.1.1) — the Params
/// default; the lower orders exist for [.diag] quadrature-sensitivity
/// probes.
[[nodiscard]] QuadratureRule quadrature_rule_from_count(int n)
{
    switch (n) {
    case 1:  return QuadratureRule::OnePointCentroid;
    case 3:  return QuadratureRule::ThreePointEdgeMid;
    case 7:  return QuadratureRule::SevenPointDunavant;
    case 12: return QuadratureRule::TwelvePointDunavant;
    default:
        throw std::invalid_argument(
            "LMEAssembler: n_quadrature_per_tri must be one of "
            "{1, 3, 7, 12} (got " + std::to_string(n) + ").");
    }
}

/// Curved-shell @ref LMEAssembler::assemble_K. Despite the legacy
/// @c _curved_bending name this assembles BOTH the Kirchhoff–Love
/// bending block (k_B) and the curved-metric membrane block (k_L) from
/// the shared C_chart; a membrane-only or bending-only material is fine.
///
/// Strategy P2 quadrature (input-F triangles in @f$ \mathbb R^3 @f$):
/// at each Gauss point @c x_g we sum @f$ \sum_A w_A^Q(x_g) @f$ across
/// the Shepard-active patches, project @c x_g into each chart via
/// @c lme::Patch::V and the chart's centroid, evaluate the 2D LME
/// Hessian, and accumulate the bending stiffness block.
///
/// Bending strain on a Kirchhoff–Love shell is
/// @f$ \rho_{\alpha\beta} = -t_0 \cdot u_{,\alpha\beta} -
///                          \varphi_{0,\alpha\beta} \cdot \Delta t(u) @f$
/// (Millán 2011 §3.3, eq. 14). Both terms ship: the first is exact on
/// planar meshes (the chart normal @f$ t_0 = V_A^{(1)} \times V_A^{(2)} @f$
/// collapses to @f$ \pm \hat z @f$ so the per-(a, b) 3x3 block reduces
/// to a single z-z diagonal entry); the second vanishes identically
/// on planar meshes (the chart Hessian sum
/// @f$ \varphi_{0,\alpha\beta} = \sum_b \partial_\alpha\partial_\beta
/// p_b(\xi_g) \cdot P_b = 0 @f$ because LME's linear reproduction
/// forces @f$ \sum_b p_b(\xi) \cdot P_b = (\xi, 0) @f$, whose second
/// chart-derivative is zero). The curvature-coupling piece picks up
/// on weakly-curved shells via @f$ \Delta t(u) = j_0^{-1} (I - t_0
/// t_0^T) X(u) @f$ with @f$ X(u_b) = \partial_1 p_b (u_b \times
/// \varphi_{0,2}) + \partial_2 p_b (\varphi_{0,1} \times u_b) @f$ —
/// closing the §10 step 9.7b deliverable.
/// Shared node-set + chart construction for the curved LME assembly paths.
/// @ref assemble_K_curved_bending, @ref assemble_M_curved and
/// @ref LMEAssembler::evaluate_modes_at_vertices all build the SAME
/// ghost-augmented node set, per-node β vectors, value-based @c r_cut,
/// vertex adjacency and per-vertex max-ent charts. Factored here so the
/// three sites cannot drift; the K / M gates validate that the assembled
/// matrices are byte-unchanged by the extraction.
struct CurvedPatchContext {
    std::vector<lme::BoundaryEdge> bdry;
    Eigen::MatrixXd  V_ext;          ///< nodes incl. ghosts, (n_ext × 3)
    Eigen::VectorXd  h_a_ext;        ///< per-node one-ring spacing
    Eigen::Index     n_v   = 0;      ///< real vertices
    Eigen::Index     n_ext = 0;      ///< real + ghost nodes
    Eigen::VectorXd  beta_lme;
    Eigen::VectorXd  beta_wpca;
    Eigen::VectorXd  beta_patches;
    double           r_cut = 0.0;
    std::vector<std::vector<int>> adjacency;
    std::vector<lme::Patch> patches;
    Eigen::MatrixXd  patch_points;   ///< = V (real-vertex PoU centres)
};

[[nodiscard]] CurvedPatchContext build_curved_patch_context(
    const Eigen::MatrixXd&      V,
    const Eigen::MatrixXi&      F,
    const LMEAssembler::Params& params)
{
    CurvedPatchContext c;
    const Eigen::Index n_v = V.rows();
    c.n_v = n_v;

    // ----- locality parameters --------------------------------------
    const Eigen::VectorXd h_a_real = vertex_one_ring_h(V, F);

    // ----- ghost-node augmentation (Millán 2011 §4.1.2): reflect one
    // ghost per boundary edge so per-Gauss-point Newton-in-chart keeps
    // node support past the rim.
    c.V_ext   = V;
    c.h_a_ext = h_a_real;
    if (params.use_ghost_nodes) {
        c.bdry = lme::collect_boundary_edges(F);
        if (!c.bdry.empty()) {
            const Eigen::MatrixXd ghosts =
                lme::build_ghost_positions(V, F, c.bdry);
            const Eigen::Index G = ghosts.rows();
            c.V_ext.conservativeResize(n_v + G, 3);
            c.V_ext.bottomRows(G) = ghosts;
            c.h_a_ext.conservativeResize(n_v + G);
            for (Eigen::Index i = 0; i < G; ++i) {
                const auto& b = c.bdry[static_cast<std::size_t>(i)];
                c.h_a_ext(n_v + i) =
                    0.5 * (h_a_real(b.v0) + h_a_real(b.v1));
            }
        }
    }
    const Eigen::Index n_ext = c.V_ext.rows();
    c.n_ext = n_ext;

    // beta_lme: per-node (ghosts included); beta_patches: per-PoU-centre
    // (real vertices only — ghosts never anchor a patch).
    c.beta_lme.resize(n_ext);
    c.beta_wpca.resize(n_ext);
    c.beta_patches.resize(n_v);
    for (Eigen::Index a = 0; a < n_ext; ++a) {
        const double h2 = c.h_a_ext(a) * c.h_a_ext(a);
        c.beta_lme(a)   = params.gamma      / h2;
        c.beta_wpca(a)  = params.gamma_wpca / h2;
        if (a < n_v) {
            c.beta_patches(a) = params.gamma_pu / h2;
        }
    }

    // Value-based neighbour cutoff (Millán 2011 Eq. 2); the SME branch uses
    // the paper's γ_eff = 2/α (closing C7). Identical for K and M so they
    // live on the same discrete function space.
    c.r_cut =
        params.use_second_order_sme
            ? std::sqrt(-std::log(params.tol_lme)
                        * params.sme_alpha / 2.0)
                  * c.h_a_ext.maxCoeff()
            : std::sqrt(-std::log(params.tol_lme) / params.gamma)
                  * c.h_a_ext.maxCoeff();

    // Vertex adjacency, augmented with ghost-to-(v0,v1) edges so k-ring BFS
    // from real anchors reaches the ghosts.
    c.adjacency = build_vertex_adjacency(n_v, F);
    if (params.use_ghost_nodes && !c.bdry.empty()) {
        c.adjacency.resize(static_cast<std::size_t>(n_ext));
        for (std::size_t i = 0; i < c.bdry.size(); ++i) {
            const auto& b = c.bdry[i];
            const int   g = static_cast<int>(n_v) + static_cast<int>(i);
            c.adjacency[static_cast<std::size_t>(b.v0)].push_back(g);
            c.adjacency[static_cast<std::size_t>(b.v1)].push_back(g);
            c.adjacency[static_cast<std::size_t>(g)].push_back(b.v0);
            c.adjacency[static_cast<std::size_t>(g)].push_back(b.v1);
        }
        std::unordered_set<int> touched;
        for (const auto& b : c.bdry) {
            touched.insert(b.v0);
            touched.insert(b.v1);
        }
        for (int v : touched) {
            auto& nbrs = c.adjacency[static_cast<std::size_t>(v)];
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }

    // Per-vertex charts: connected metric-ball for 1st-order LME, flat
    // topological k-ring for SME (its moment constraints are projected-
    // position sensitive). See the assemble_K_curved_bending notes.
    const int    kRingDepth      = params.k_ring_depth;
    const double lme_chart_factor =
        std::sqrt(-std::log(params.chart_tol_lme) / params.gamma);
    auto build_neighbours = [&](int a) -> std::vector<int> {
        if (params.use_second_order_sme) {
            if (params.sme_chart_radius_mult > 0.0) {
                return lme::extract_chart_neighbourhood_intrinsic(
                    a, params.sme_chart_radius_mult * c.h_a_ext(a),
                    c.V_ext, c.adjacency,
                    /*min_nodes=*/7, params.max_chart_nodes);
            }
            return k_ring(a, kRingDepth, c.adjacency);
        }
        return lme::extract_chart_neighbourhood(
            a, lme_chart_factor * c.h_a_ext(a), c.V_ext, c.adjacency,
            /*min_nodes=*/7, params.max_chart_nodes);
    };
    c.patches.reserve(static_cast<std::size_t>(n_v));
    for (Eigen::Index a = 0; a < n_v; ++a) {
        c.patches.push_back(lme::build_patch(
            static_cast<int>(a), c.V_ext,
            build_neighbours(static_cast<int>(a)), c.beta_wpca));
    }
    c.patch_points = V;
    return c;
}

[[nodiscard]] Eigen::SparseMatrix<double> assemble_K_curved_bending(
    const Eigen::MatrixXd&                       V,
    const Eigen::MatrixXi&                       F,
    const ShellMaterial&                         material,
    const LMEAssembler::Params&                  params,
    LMEAssembler::DropStats*                     drop_stats_out = nullptr)
{
    // Reset the drop-stats sink at the START (the header contract), not just
    // on the success path at the end. Otherwise a throw below (a drop-fraction
    // abort or the allFinite backstop) would leave last_drop_stats() reporting
    // the previous successful call's figures.
    if (drop_stats_out != nullptr) {
        *drop_stats_out = LMEAssembler::DropStats{};
    }
    if (V.cols() != 3) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_K (curved): V must have exactly 3 columns");
    }
    const double D_bend = material.k_B;
    const double k_mem  = material.k_L;
    if (!(D_bend > 0.0) && !(k_mem > 0.0)) {
        return Eigen::SparseMatrix<double>(3 * V.rows(), 3 * V.rows());
    }

    const Eigen::Index n_v = V.rows();
    const double       nu  = material.poisson_ratio;

    // Shared node-set + chart construction (also used by assemble_M_curved
    // and LMEAssembler::evaluate_modes_at_vertices). Aliased below so the
    // per-Gauss-point assembly code downstream reads exactly as before.
    const CurvedPatchContext _ctx = build_curved_patch_context(V, F, params);
    const auto&        bdry         = _ctx.bdry;
    const auto&        V_ext        = _ctx.V_ext;
    const auto&        h_a_ext      = _ctx.h_a_ext;
    const Eigen::Index n_ext        = _ctx.n_ext;
    const auto&        beta_lme     = _ctx.beta_lme;
    const auto&        beta_patches = _ctx.beta_patches;
    const double       r_cut        = _ctx.r_cut;
    const auto&        adjacency    = _ctx.adjacency;
    const auto&        patches      = _ctx.patches;
    const auto&        patch_points = _ctx.patch_points;

    // Per-patch neighbour membership set for the "patch A owns triangle t"
    // test (Millán 2011 §4.1.1 faithful per-patch quadrature). The paper
    // integrates each patch's energy E_A over a triangulation of THAT
    // patch's own parametric domain Λ_A, so every quadrature point lies
    // inside the patch's chart node hull by construction. We realise this
    // on the input mesh (itself a valid surface triangulation): patch A may
    // only integrate a mesh triangle ALL of whose vertices lie in A's chart
    // neighbourhood — then the triangle, and hence every Gauss point on it,
    // projects strictly inside conv(pA.xi) and the in-chart max-ent solve is
    // feasible. Non-owning patches are excluded from the PoU at that Gauss
    // point and the surviving weights renormalised. This replaces the
    // previous cross-projection of a global Gauss point into EVERY Shepard-
    // active chart, which let points land outside a foreign chart's hull →
    // genuine linear infeasibility (the anisotropic-cylinder SME divergence).
    std::vector<std::unordered_set<int>> patch_nbr_set(
        static_cast<std::size_t>(patches.size()));
    for (std::size_t A = 0; A < patches.size(); ++A) {
        const auto& nids = patches[A].neighbor_ids;
        patch_nbr_set[A].reserve(nids.size() * 2);
        patch_nbr_set[A].insert(nids.begin(), nids.end());
    }

    // INTERIOR ownership for the 2nd-order SME path (2026-06-07,
    // rim-infeasibility root cause): SME's moment-with-gap equality
    // needs BALANCED node coverage around the query, not just hull
    // membership — at the chart's coverage fringe the cloud is
    // one-sided and the program is GENUINELY infeasible (certified by
    // the separating-hyperplane dissection, [sme_rim_dissect]; one
    // extra ring of surround restores feasibility, and gap enlargement
    // makes it strictly worse). The paper never meets this case: M11
    // sizes patch neighbourhoods by the BASIS tolerance (γ_LME = 0.8 ⇒
    // ~5.4 h) while the PU quadrature reach is γ_PU ∈ [4,6] ⇒ ~1.9 h —
    // every query sits several spacings inside its patch's coverage by
    // construction. Our SME chart is the FLAT k-ring (C4, ~1.4–3 h, to
    // avoid arc-folding of the position-sensitive moment constraints),
    // i.e. SMALLER than the PU reach, so we restore the paper's
    // invariant on the quadrature side instead: a patch integrates a
    // triangle only if the triangle's vertices AND THEIR FULL
    // (ghost-augmented) 1-RINGS lie in its chart. The triangle's own
    // vertex-anchored patches always qualify (1-ring extension ⊆
    // 2-ring ⊆ k-ring-3 chart), so at least three owners survive at
    // every Gauss point — same coverage guarantee as the plain filter,
    // decided by geometry upfront rather than by Newton failure at
    // run time. 1st-order LME keeps the plain vertex filter: its
    // feasibility condition is hull membership only, which the plain
    // filter already guarantees.
    std::vector<std::unordered_set<int>> patch_interior_set;
    if (params.use_second_order_sme) {
        patch_interior_set.resize(patches.size());
        for (std::size_t A = 0; A < patches.size(); ++A) {
            const auto& nbr = patch_nbr_set[A];
            auto&       out = patch_interior_set[A];
            out.reserve(nbr.size());
            for (int v : patches[A].neighbor_ids) {
                if (v < 0
                    || v >= static_cast<int>(adjacency.size()))
                    continue;
                bool interior = true;
                for (int u :
                     adjacency[static_cast<std::size_t>(v)]) {
                    if (!nbr.count(u)) { interior = false; break; }
                }
                if (interior) out.insert(v);
            }
        }
    }

    // ----- 2nd-order SME nodal-gap matrices (paper §3.2) ------------
    // Per-patch list of per-chart-node 2x2 PSD slack matrices d_a,
    // consumed by the SME inner-loop dispatch in Phase B.1c. Only
    // built when use_second_order_sme is on; the LME-only path
    // stays zero overhead.
    std::vector<std::vector<Eigen::Matrix2d>> sme_gaps_per_patch;
    if (params.use_second_order_sme) {
        // bdry was only collected on the ghost-on path; the SME
        // classification needs the global boundary regardless.
        std::vector<lme::BoundaryEdge> bdry_owned;
        const std::vector<lme::BoundaryEdge>* bdry_for_sme = &bdry;
        if (!params.use_ghost_nodes) {
            bdry_owned    = lme::collect_boundary_edges(F);
            bdry_for_sme  = &bdry_owned;
        }
        const std::vector<Eigen::Matrix3d> node_cov =
            vertex_one_ring_covariance(V, F);
        const std::vector<std::vector<Eigen::Vector3d>> one_ring_edges =
            vertex_one_ring_edges(V, F);
        const std::vector<VtxBoundaryFrame> sme_frames =
            compute_sme_boundary_frames(
                V, *bdry_for_sme, adjacency, n_v);
        sme_gaps_per_patch = build_per_patch_sme_gaps(
            patches, sme_frames, n_v, h_a_ext, node_cov, one_ring_edges,
            params.sme_alpha, params.sme_beta);
    }

    // The Kirchhoff bending constitutive matrix is built per Gauss point
    // from the chart's contravariant metric a^{αβ} (C_bend_chart, in the
    // bending block below) — metric-dependent exactly like the membrane
    // D_mem_chart, per Millán 2011 §3.4 — rather than a single hard-coded
    // identity-metric matrix, which is correct only on flat/orthonormal
    // charts.

    const auto qpts = chladni::shell::quadrature_points(
        quadrature_rule_from_count(params.n_quadrature_per_tri));

    // ----- triplet accumulation ------------------------------------
    // Per-Gauss-point block accumulator: collapses the ~|active
    // Shepard patches| × n_act² duplicate (a, b) contributions into a
    // single 3×3 block per (a, b) pair before emitting triplets. On
    // the icosphere k=2 fixture this drops global triplet emissions
    // from ~100 M (raw) to ~500 k (deduplicated), eliminating the
    // malloc-thrashing wall that previously gated the [.slow] tags
    // on the 8×8 plate and icosphere tests.
    //
    // Key packs (a, b) into a single 64-bit int so std::unordered_map
    // can hash directly without an extra pair wrapper. Bound on n_v
    // here is the typical mesh-vertex count (≤ 2^31), which keeps
    // (a, b) within the 64-bit range comfortably.
    // Parallel triangle loop. Each worker thread owns its own
    // accumulators (gauss_blocks/thread_blocks) and triplet buffer; the
    // per-thread buffers are concatenated into a single triplet vector
    // for setFromTriplets at the end. The F-loop is embarrassingly
    // parallel modulo the triplet writes — the per-triangle math reads
    // V, F, patches, and the β / r_cut constants (all
    // read-only) and writes only to thread-local state. Spectra
    // duplicate triplets correctly merge in setFromTriplets so we do
    // not need cross-thread dedup at the (a, b) level.
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_threads_raw = static_cast<int>(hw == 0 ? 1u : hw);
    const Eigen::Index n_F = F.rows();
    const int n_threads = static_cast<int>(
        std::min<Eigen::Index>(n_threads_raw, std::max<Eigen::Index>(1, n_F)));
    std::vector<std::vector<Eigen::Triplet<double>>> trips_per_thread(
        static_cast<std::size_t>(n_threads));

    // ----- LME_DIAG drop accounting (env-gated, zero-cost when off) --
    // Per-thread tallies of the two silent PoU-drop channels — the
    // §4.1.1 ownership filter and the Newton-failure catch — plus a
    // per-anchor Newton-failure histogram, so a mesh where the rim
    // machinery quietly sheds stiffness (e.g. the QuadSplit-rim bug)
    // is visible instead of silently renormalised away.
    const bool diag_drops = std::getenv("LME_DIAG") != nullptr;
    // First-N full dumps of FAILING SME (chart, Gauss point) instances —
    // chart node positions, gap matrices, query — so a failure can be
    // reproduced standalone and its separating direction dissected
    // (rim-infeasibility root-cause arc, 2026-06-07). Env-gated with
    // the rest of the diagnostics; prints chart-PHYSICAL (unscaled)
    // data plus h_chart so the dissection applies the same h_chart=1
    // rescale the assembler uses.
    std::atomic<int> sme_fail_dumps{0};
    constexpr int kSmeFailDumpMax = 3;
    struct DropTally {
        long   n_own       = 0;
        long   n_newton    = 0;
        long   n_escal_ok  = 0;  ///< SME evals rescued by α escalation
        long   escal_steps = 0;  ///< total ×2 steps spent on rescues
        double w_own       = 0.0;
        double w_newton    = 0.0;
        /// Per-GAUSS-POINT dropped-weight distribution (w_failed_sum at
        /// the renormalisation site — the fraction of that point's PoU
        /// shed before rescaling). Discriminates diffuse drops (max ~
        /// 1e-4, harmless renormalisation noise) from concentrated ones
        /// (a Gauss point losing a third of its partition).
        double w_gauss_max = 0.0;
        long   n_gauss_gt001 = 0;  ///< Gauss points with w_failed > 0.01
        long   n_gauss_gt01  = 0;  ///< … > 0.1
        long   n_gauss_gt03  = 0;  ///< … > 0.3
        std::unordered_map<int, long> newton_by_anchor;
        /// Per-anchor α-escalation rescues (diag only) — locates WHERE
        /// the B3 ladder fires (e.g. the geodesic-hemisphere pinch
        /// corners), the way newton_by_anchor locates final drops.
        std::unordered_map<int, long> escal_by_anchor;
    };
    std::vector<DropTally> drops_per_thread(
        static_cast<std::size_t>(n_threads));

    auto block_key = [n_ext_l = static_cast<std::int64_t>(n_ext)](
                          int a, int b) -> std::int64_t {
        return static_cast<std::int64_t>(a) * n_ext_l
             + static_cast<std::int64_t>(b);
    };

    auto worker = [&](int tid) {
        std::unordered_map<std::int64_t, Eigen::Matrix3d> gauss_blocks;
        // Per-THREAD (a, b) accumulator, persistent across this thread's whole
        // triangle range. Each (a, b) basis interaction is integrated over
        // every triangle in the two functions' shared support (~tens of them
        // on a wide LME/SME stencil); folding all of them into one 3×3 block
        // before emitting triplets — instead of emitting one block per
        // triangle — is what keeps the triplet stream O(nnz) rather than
        // O(nnz × support-triangles). On icosphere k=4 this is the difference
        // between ~2.9 M and ~196 M emitted triplets (a 6.3 GB → 0.1 GB peak).
        std::unordered_map<std::int64_t, Eigen::Matrix3d> thread_blocks;
        gauss_blocks.reserve(512);
        thread_blocks.reserve(4096);

        // Thread-local scratch for the per-node 3×3 strain-displacement
        // blocks and their constitutive products. Hoisted out of the
        // Gauss-point loop and reused via resize() (capacity grows to the
        // largest chart once, then no further allocation) — the four
        // std::vector<Matrix3d> were otherwise heap-allocated afresh at
        // every Gauss point of every patch.
        std::vector<Eigen::Matrix3d> Bb_s, CBb_s, Bm_s, DBm_s;
        auto& trips = trips_per_thread[static_cast<std::size_t>(tid)];
        trips.reserve(
            static_cast<std::size_t>(3 * n_ext) * 9 * 32
            / static_cast<std::size_t>(n_threads));

        auto accumulate_block = [&](int a, int b,
                                     const Eigen::Matrix3d& block) {
            const std::int64_t key = block_key(a, b);
            auto it = gauss_blocks.find(key);
            if (it == gauss_blocks.end()) {
                gauss_blocks.emplace(key, block);
            } else {
                it->second += block;
            }
        };

        const Eigen::Index chunk = (n_F + n_threads - 1) / n_threads;
        const Eigen::Index t_start = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index t_end   = std::min(t_start + chunk, n_F);

        for (Eigen::Index t = t_start; t < t_end; ++t) {
        // NB: thread_blocks is NOT cleared here — it accumulates every
        // triangle's (a, b) contributions for this thread; emitted once
        // after the loop.
        const int i0 = F(t, 0), i1 = F(t, 1), i2 = F(t, 2);
        const Eigen::Vector3d P0 = V.row(i0).transpose();
        const Eigen::Vector3d P1 = V.row(i1).transpose();
        const Eigen::Vector3d P2 = V.row(i2).transpose();

        // Triangle area in R³ via the cross product. For a planar
        // mesh this equals |signed_double_area_2d| exactly (the cross
        // product magnitude is the 2D determinant when the triangle
        // lies in the xy-plane), so the flat-plate regression is
        // preserved at the quadrature-weight level.
        const Eigen::Vector3d e01 = P1 - P0;
        const Eigen::Vector3d e02 = P2 - P0;
        const double          two_area = e01.cross(e02).norm();

        for (const auto& q : qpts) {
            const double u_bary = 1.0 - q.v - q.w;
            const Eigen::Vector3d Pq =
                u_bary * P0 + q.v * P1 + q.w * P2;

            const double w_dA = q.weight * two_area;

            // PoU truncation = kPoUTruncTol (shared with the mass path so
            // K and M stay on the same function space).
            const double tol_pu = kPoUTruncTol;
            const lme::ShepardWeights sw = lme::shepard_partition(
                patch_points, beta_patches, Pq, tol_pu);

            gauss_blocks.clear();
            // Per-Gauss-point bookkeeping for the catch-and-
            // renormalise safety net (see comment at the catch
            // block below). MEASURED 2026-06-04 (LME_DIAG over the
            // default + slow suites): the 1st-order LME path never
            // fires the Newton-drop channel — Q-D1 ownership + Q-D6
            // geometric support keep its queries feasible (the
            // earlier "72x8+ kRing-reach gap" rationale is
            // obsolete). The live consumer is the SME path, which
            // drops marginal-weight patches routinely (B.2 disk bar
            // w≈2e-5 per Gauss point; lat-long hemisphere levels up
            // to w=0.52 cumulative), so the net stays on both paths.
            double w_failed_sum = 0.0;
            // Newton-failure weight only (the ownership-filter exclusions
            // below are routine, not failures); budgeted against
            // params.max_newton_drop_frac at the renormalisation site.
            double w_newton_sum = 0.0;

            for (std::size_t s = 0; s < sw.indices.size(); ++s) {
                const int                A    = sw.indices[s];
                const double             w_A  = sw.values[s];
                const lme::Patch&        pA   = patches[
                    static_cast<std::size_t>(A)];

                // Faithful per-patch quadrature (Millán 2011 §4.1.1): patch
                // A integrates this triangle only if it OWNS it — all three
                // vertices lie in A's chart neighbourhood — so the Gauss
                // point projects inside conv(pA.xi) and the in-chart solve
                // is feasible. The triangle's own vertex-patches always own
                // it (their 1-ring ⊇ {i0,i1,i2}) and carry the dominant PoU
                // weight, so at least three owners always survive. Non-owners
                // are removed from the PoU here and renormalised via
                // w_failed_sum below.
                // SME: INTERIOR ownership (vertices + their 1-rings in
                // the chart) — see the patch_interior_set comment.
                // LME: plain vertex ownership (hull membership is its
                // only feasibility requirement).
                const auto& ownA =
                    params.use_second_order_sme
                        ? patch_interior_set[static_cast<std::size_t>(A)]
                        : patch_nbr_set[static_cast<std::size_t>(A)];
                if (!ownA.count(i0) || !ownA.count(i1) || !ownA.count(i2)) {
                    w_failed_sum += w_A;
                    {
                        auto& dt = drops_per_thread[
                            static_cast<std::size_t>(tid)];
                        ++dt.n_own;
                        dt.w_own += w_A;
                    }
                    continue;
                }

                // Chart projection of the Gauss point.
                const Eigen::Vector3d delta_g = Pq - pA.Qbar;
                Eigen::Vector2d       xi_g    = pA.V.transpose() * delta_g;

                // Per-neighbour β slice for the in-chart 2D solve.
                const Eigen::Index n_neigh =
                    static_cast<Eigen::Index>(pA.neighbor_ids.size());
                Eigen::VectorXd beta_chart(n_neigh);
                for (Eigen::Index k = 0; k < n_neigh; ++k) {
                    beta_chart(k) = beta_lme(
                        pA.neighbor_ids[static_cast<std::size_t>(k)]);
                }

                // The in-chart Newton on a wPCA-projected k-ring
                // neighbourhood can fail at Gauss points where the
                // chart geometry is degenerate (chart's node cloud
                // collapses through a high-valence vertex; or the
                // Gauss point projects outside the in-chart convex
                // hull at the rim of an open-edge mesh, beyond
                // ghost extension on coarse rings). The failing
                // patches carry marginal Shepard weight at those
                // Gauss points — MEASURED 2026-06-06 (per-Gauss
                // dropped-weight distribution, LME_DIAG): max 0.2 %
                // of a single Gauss point's PoU weight on the worst
                // fixture (aspect-2.39 cylinder), ≤1e-5 typical —
                // so dropping the offending patch and renormalising
                // the survivors perturbs K negligibly. NOT a design
                // guarantee, only an observation: the per-Gauss
                // distribution is the watchdog, and a principled
                // replacement for this net remains an open item
                // (user-flagged 2026-06-06).
                //
                // The catch is also REQUIRED for thread safety:
                // assemble_K_curved_bending is parallelised across
                // F, and a propagating exception from a worker
                // thread aborts the process via std::terminate
                // (the OpenMP runtime doesn't tunnel exceptions
                // back to the master).
                LMEBasisGradHess gh;
                try {
                    if (params.use_second_order_sme) {
                        // SME's Newton on (λ, μ) ∈ R^5 packs the
                        // dual residual rows with magnitudes O(xi)
                        // (λ rows) and O(xi²) (μ rows); the absolute
                        // infinity-norm convergence check then takes
                        // h_chart² relative accuracy on the μ rows.
                        // For fixtures with h_chart << 1 (typical on
                        // physical-units geometries: R=0.1 m, h~0.01
                        // m) Newton stalls before reaching the tol.
                        // Renormalise the chart to h_chart=1 — xi'
                        // = xi/h_chart, d' = d/h_chart² — so the SME
                        // dual lives in O(1) and the absolute tol
                        // reads as relative. Basis values s_a are
                        // dimensionless under this transform (the
                        // Gibbs normaliser cancels the rescaling),
                        // so the returned values are exact; the
                        // returned gradients and Hessians come back
                        // in the SCALED chart's units and are
                        // EXPLICITLY rescaled to chart-physical
                        // units below (×1/h_chart, ×1/h_chart²)
                        // before the strain blocks consume them.
                        // (An earlier comment claimed the factors
                        // "cancel by dimensional analysis" — wrong,
                        // audit item D9/M-D3; the explicit
                        // back-scaling is what makes the transform
                        // value-faithful.)
                        const int sme_iters_cap = std::max(
                            params.newton_max_iters,
                            5 * params.newton_max_iters / 2);
                        const double h_chart =
                            h_a_ext(pA.anchor_id);
                        const double inv_h = 1.0 / h_chart;
                        const Eigen::MatrixXd xi_scaled =
                            pA.xi * inv_h;
                        const Eigen::Vector2d xi_g_scaled =
                            xi_g * inv_h;
                        std::vector<Eigen::Matrix2d> d_scaled =
                            sme_gaps_per_patch[
                                static_cast<std::size_t>(A)];
                        const double inv_h2 = inv_h * inv_h;
                        for (auto& dk : d_scaled) dk *= inv_h2;
                        // Closed-form Hessian (App. C): exact, and ~2.1x
                        // faster per call than the FD-on-gradient path
                        // ([.diag][diag_sme_timing]: 8.2 vs 17.5 us) —
                        // one Newton solve instead of 1 + 4 perturbed.
                        // It also carries no active-set-match constraint
                        // between perturbed queries, so it never trips
                        // the FD path's "active set changes" throw near
                        // the chart-support boundary. The FD path remains
                        // the [.diag] cross-check reference.
                        //
                        // Per-patch α ESCALATION
                        // (Params::sme_alpha_escalation_steps): a
                        // failure here is retried with the patch's
                        // gap matrices enlarged ×2 per step
                        // (α_eff = α·2^k) before falling through to
                        // the drop-net — gap enlargement is the
                        // paper's own feasibility mechanism (RMA13
                        // §3.2–3.3), and locally enlarged slack is
                        // strictly gentler than dropping the patch
                        // from the PoU. Rescues the marginal
                        // infeasibility the 12-pt quadrature exposes
                        // on coarse curved meshes (geodesic
                        // hemisphere at the paper's α=2).
                        LMEBasisGradHess gh_scaled;
                        for (int esc = 0;; ++esc) {
                            try {
                                gh_scaled = lme::
                                    evaluate_sme_basis_grad_and_hess_closed_form(
                                        xi_scaled, d_scaled,
                                        xi_g_scaled, r_cut * inv_h,
                                        params.newton_tol,
                                        sme_iters_cap);
                                if (esc > 0) {
                                    auto& dt = drops_per_thread[
                                        static_cast<std::size_t>(tid)];
                                    ++dt.n_escal_ok;
                                    dt.escal_steps += esc;
                                    if (diag_drops)
                                        ++dt.escal_by_anchor[
                                            pA.anchor_id];
                                }
                                break;
                            } catch (const std::runtime_error&) {
                                if (esc >= std::max(
                                        0,
                                        params
                                            .sme_alpha_escalation_steps))
                                    throw;
                                for (auto& dk : d_scaled) dk *= 2.0;
                            }
                        }
                        // Rescale gradients (1/h_chart) and Hessians
                        // (1/h_chart²) back to chart-physical units.
                        gh.indices = gh_scaled.indices;
                        gh.values  = gh_scaled.values;
                        gh.gradients.resize(gh_scaled.gradients.size());
                        gh.hessians.resize(gh_scaled.hessians.size());
                        for (std::size_t k = 0;
                             k < gh_scaled.gradients.size(); ++k)
                        {
                            gh.gradients[k] =
                                gh_scaled.gradients[k] * inv_h;
                            gh.hessians[k] =
                                gh_scaled.hessians[k] * inv_h2;
                        }
                    } else {
                        gh = lme::evaluate_basis_grad_and_hess(
                            pA.xi, beta_chart, xi_g, r_cut,
                            params.newton_tol, params.newton_max_iters);
                    }
                } catch (const std::runtime_error&) {
                    w_failed_sum += w_A;
                    w_newton_sum += w_A;  // counts toward the abort budget
                    {
                        auto& dt = drops_per_thread[
                            static_cast<std::size_t>(tid)];
                        ++dt.n_newton;
                        dt.w_newton += w_A;
                        if (diag_drops)
                            ++dt.newton_by_anchor[pA.anchor_id];
                        if (diag_drops
                            && params.use_second_order_sme
                            && sme_fail_dumps.fetch_add(1)
                                   < kSmeFailDumpMax) {
                            const auto& gaps = sme_gaps_per_patch[
                                static_cast<std::size_t>(A)];
                            std::fprintf(stderr,
                                "[sme_fail_dump] anchor=%d "
                                "gauss_xi=(%.17g %.17g) h_chart=%.17g "
                                "w_A=%.3g n_chart=%zu\n",
                                pA.anchor_id, xi_g(0), xi_g(1),
                                h_a_ext(pA.anchor_id), w_A,
                                pA.neighbor_ids.size());
                            for (std::size_t k = 0;
                                 k < pA.neighbor_ids.size(); ++k) {
                                const int gid = pA.neighbor_ids[k];
                                std::fprintf(stderr,
                                    "[sme_fail_dump]   k=%zu gid=%d "
                                    "ghost=%d xi=(%.17g %.17g) "
                                    "d=(%.17g %.17g %.17g)\n",
                                    k, gid,
                                    gid >= static_cast<int>(n_v) ? 1 : 0,
                                    pA.xi(static_cast<Eigen::Index>(k), 0),
                                    pA.xi(static_cast<Eigen::Index>(k), 1),
                                    gaps[k](0, 0), gaps[k](1, 1),
                                    gaps[k](0, 1));
                            }
                        }
                    }
                    continue;
                }

                const Eigen::Index n_act =
                    static_cast<Eigen::Index>(gh.indices.size());

                // wPCA frame normal — used **only** as a sign reference
                // for the dynamic surface normal below; not used in the
                // strain measure directly.
                const Eigen::Vector3d t0_chart = pA.V.col(0).cross(pA.V.col(1));

                // Shared per-Gauss-point surface geometry, computed ONCE
                // and consumed by BOTH the bending and membrane blocks:
                // the reference-surface tangents
                //   φ_{0,α}(x_g) = Σ_b ∂_α p_b(ξ_g) · P_b,
                // and the Kirchhoff–Love constitutive Voigt matrix
                // C(a^{αβ}) built from the inverse first fundamental form
                // a^{αβ} = (φ_α·φ_β)^{-1}. Per Millán 2011 §3.4 the SAME
                // C governs membrane (n = h C ε) and bending
                // (m = (h³/12) C ρ), so it is assembled once here instead
                // of twice — the bending and membrane blocks previously
                // each re-ran the O(n_act) tangent gather + an identical
                // metric inverse + an identical C build every Gauss point.
                Eigen::Vector3d phi_1 = Eigen::Vector3d::Zero();
                Eigen::Vector3d phi_2 = Eigen::Vector3d::Zero();
                Eigen::Matrix3d C_chart;
                if (D_bend > 0.0 || k_mem > 0.0) {
                    for (Eigen::Index k = 0; k < n_act; ++k) {
                        const int    lid = gh.indices[
                            static_cast<std::size_t>(k)];
                        const int    gid = pA.neighbor_ids[
                            static_cast<std::size_t>(lid)];
                        const Eigen::VectorXd& dp = gh.gradients[
                            static_cast<std::size_t>(k)];
                        phi_1 += dp(0) * V_ext.row(gid).transpose();
                        phi_2 += dp(1) * V_ext.row(gid).transpose();
                    }
                    Eigen::Matrix2d a_mat;
                    a_mat(0, 0) = phi_1.dot(phi_1);
                    a_mat(0, 1) = phi_1.dot(phi_2);
                    a_mat(1, 0) = a_mat(0, 1);
                    a_mat(1, 1) = phi_2.dot(phi_2);
                    // Degenerate-chart guard: a_mat is the first
                    // fundamental form (Gram matrix of the reference-
                    // surface tangents), so det(a_mat) = |φ_1 × φ_2|² is
                    // the squared area element and a_det/a_scale = sin²θ
                    // between the tangents. It collapses to zero at a
                    // degenerate Gauss point (pinched chart, sliver
                    // triangle, collinear/zero tangents). Inverting a
                    // singular a_mat returns inf/NaN that would propagate
                    // silently through C_chart → strain blocks → triplets
                    // (the emitter filters on v != 0, not finiteness) and
                    // surface only as an opaque downstream eigensolver
                    // failure. Surface it here as a located error instead,
                    // matching the j0 > 0 discipline below.
                    const double a_det =
                        a_mat(0, 0) * a_mat(1, 1)
                        - a_mat(0, 1) * a_mat(1, 0);
                    const double a_scale = a_mat(0, 0) * a_mat(1, 1);
                    if (!std::isfinite(a_det) || !(a_scale > 0.0)
                        || a_det <= 1.0e-10 * a_scale) {
                        throw std::runtime_error(
                            "LMEAssembler::assemble_K (curved): degenerate "
                            "surface metric at a quadrature point of "
                            "triangle " + std::to_string(t) + " (patch "
                            + std::to_string(A) + ") — first fundamental "
                            "form det " + std::to_string(a_det)
                            + " collapsed relative to scale "
                            + std::to_string(a_scale)
                            + " (parallel/zero reference tangents). The "
                            "metric inverse would inject a NaN into K; "
                            "remesh to remove the degenerate chart.");
                    }
                    const Eigen::Matrix2d a_inv = a_mat.inverse();
                    const double a11 = a_inv(0, 0);
                    const double a22 = a_inv(1, 1);
                    const double a12 = a_inv(0, 1);
                    C_chart(0, 0) = a11 * a11;
                    C_chart(1, 1) = a22 * a22;
                    C_chart(0, 1) = nu * a11 * a22 + (1.0 - nu) * a12 * a12;
                    C_chart(1, 0) = C_chart(0, 1);
                    C_chart(0, 2) = a11 * a12;
                    C_chart(2, 0) = C_chart(0, 2);
                    C_chart(1, 2) = a22 * a12;
                    C_chart(2, 1) = C_chart(1, 2);
                    C_chart(2, 2) =
                        nu * a12 * a12
                        + 0.5 * (1.0 - nu) * (a11 * a22 + a12 * a12);
                }

                // ----- bending contribution -------------------------
                // Full Kirchhoff–Love bending strain (Millán 2011 §3.3
                // eq. 14):
                //   ρ_αβ = -t_0 · u_{,αβ}  -  φ_{0,αβ} · Δt(u),
                // with Δt(u) = j_0^{-1} (I - t_0 t_0^T) X(u) and
                //   X(u_b) = ∂_1 p_b (u_b × φ_{0,2})
                //          + ∂_2 p_b (φ_{0,1} × u_b).
                // On a flat plate φ_{0,αβ} = 0 identically (LME linear
                // reproduction forces the chart-Hessian sum to zero), so
                // the curvature-correction term vanishes and B_bend
                // reduces to the t_0-projected Hessian -H_voigt·t_0.
                //
                // CORRECTION (2026-05-29): this reduction is NOT
                // numerically identical to the flat path (assemble_K,
                // use_curved_shell=false). The two paths differ in the
                // BASIS STENCIL the Hessian is solved on: the flat path
                // selects all nodes within r_cut_mult·h (=4.0, wide →
                // accurate 2nd derivative); the curved path selects its
                // per-patch active set by the paper's VALUE-based
                // truncation (Millán Eq. 2 via tol_lme), sized by the
                // chart's local spacing. ERA HISTORY: before the
                // value-based cutoff landed the curved path used a fixed
                // r_cut_mult_curved=1.4 (now retired/inert), which gave
                // only ~0.64× the analytic bending energy on a structured
                // square plate (rising to ~0.97× by an effective r_cut≈3)
                // and a Scordelis-Lo roof ~22% too soft; the value-based
                // default now spans a wide stencil comparable to the flat
                // path. See tests/shell/test_static_obstacle_course.cpp.
                //
                // Implementation: build the strain-displacement matrix
                // B_bend (3 Voigt rows × 3·n_act displacement columns)
                // per active basis. The flat row is
                //   -H_voigt[s, k] · t_0^T   (1×3),
                // the curvature row is
                //   -φ_{0,αβ}^T · [Δt_b]    (1×3),
                // where [Δt_b] is the 3×3 operator acting on the
                // nodal displacement vector u_b.
                if (D_bend > 0.0) {
                    // Chart-Hessian Voigt for the flat piece.
                    Eigen::MatrixXd H_voigt(3, n_act);
                    for (Eigen::Index k = 0; k < n_act; ++k) {
                        const Eigen::MatrixXd& H = gh.hessians[
                            static_cast<std::size_t>(k)];
                        H_voigt(0, k) = H(0, 0);
                        H_voigt(1, k) = H(1, 1);
                        H_voigt(2, k) = H(0, 1);
                    }

                    // Surface tangents φ_{0,α} and the constitutive matrix
                    // C_chart are the shared per-Gauss-point quantities
                    // computed above (used here by the surface normal `t_0`
                    // and the curvature term [Δt_b], and as C·B_b below).

                    // Surface-Hessian 3-vectors per Voigt index.
                    Eigen::Vector3d phi_11 = Eigen::Vector3d::Zero();
                    Eigen::Vector3d phi_22 = Eigen::Vector3d::Zero();
                    Eigen::Vector3d phi_12 = Eigen::Vector3d::Zero();
                    for (Eigen::Index k = 0; k < n_act; ++k) {
                        const int    lid = gh.indices[
                            static_cast<std::size_t>(k)];
                        const int    gid = pA.neighbor_ids[
                            static_cast<std::size_t>(lid)];
                        const Eigen::MatrixXd& H = gh.hessians[
                            static_cast<std::size_t>(k)];
                        const Eigen::Vector3d  Pb =
                            V_ext.row(gid).transpose();
                        phi_11 += H(0, 0) * Pb;
                        phi_22 += H(1, 1) * Pb;
                        phi_12 += H(0, 1) * Pb;
                    }

                    // Surface normal at the Gauss point —
                    // t_0 = φ_{0,1} × φ_{0,2} / j_0 (Millán 2011 §3.1
                    // page 11, eq. just after (11)). On a planar mesh
                    // the LME tangents φ_{0,α} reduce to V_A.col(α)
                    // (the wPCA frame axes), so this `t_0` equals the
                    // wPCA frame normal and the flat-plate regression
                    // is preserved entrywise. On a curved mesh `t_0`
                    // follows the true surface normal at each Gauss
                    // point instead of being clamped to the patch-
                    // anchor wPCA normal — closing the dominant 60 %
                    // overshoot that the constant-`t_0` approximation
                    // caused on the sphere fixture. A sign-align step
                    // against the wPCA frame normal keeps the t_0
                    // orientation consistent across patches even when
                    // the wPCA V columns flip sign (their cross
                    // product can flip independently of the surface
                    // outward normal).
                    // j0 = |φ_1 × φ_2| = sqrt(det a_mat) > 0 is guaranteed by
                    // the degenerate-metric guard above (a_det <= 1e-10·a_scale
                    // already threw), so the surface normal t0 and the Δt
                    // projector are always well-defined — no j0 == 0 fallback
                    // is reachable here (review5/6 F6).
                    const Eigen::Vector3d phi_cross = phi_1.cross(phi_2);
                    const double          j0       = phi_cross.norm();
                    Eigen::Vector3d       t0       = phi_cross / j0;
                    if (t0.dot(t0_chart) < 0.0) {
                        t0 = -t0;
                    }

                    // Δt projector P = (I - t_0 t_0^T) / j_0.
                    const Eigen::Matrix3d P_dt =
                        (Eigen::Matrix3d::Identity() - t0 * t0.transpose())
                        / j0;

                    // Skew matrices [φ_{0,α}]_× so that
                    // [X_b] = ∂_2 p_b · [φ_{0,1}]_×
                    //       - ∂_1 p_b · [φ_{0,2}]_×.
                    auto skew = [](const Eigen::Vector3d& v) {
                        Eigen::Matrix3d S;
                        S <<  0.0, -v.z(),  v.y(),
                              v.z(),  0.0, -v.x(),
                             -v.y(),  v.x(),  0.0;
                        return S;
                    };
                    const Eigen::Matrix3d S_phi1 = skew(phi_1);
                    const Eigen::Matrix3d S_phi2 = skew(phi_2);

                    // Build the per-node 3×3 strain-displacement blocks
                    // B_b ∈ R^{3×3} (3 Voigt strain × 3 displacement),
                    // and the constitutive product C·B_b, as FIXED-SIZE
                    // Matrix3d. Fixed size matters: the per-pair products
                    // below are then fully-unrolled 3×3 kernels rather
                    // than dispatching through Eigen's dynamic gebp GEMM
                    // (which has per-call overhead that dominates at
                    // inner-dim 3 — the whole point of this rewrite).
                    std::vector<Eigen::Matrix3d>& Bb = Bb_s;
                    std::vector<Eigen::Matrix3d>& CBb = CBb_s;
                    Bb.resize(static_cast<std::size_t>(n_act));
                    CBb.resize(static_cast<std::size_t>(n_act));
                    for (Eigen::Index k = 0; k < n_act; ++k) {
                        const Eigen::VectorXd& dp = gh.gradients[
                            static_cast<std::size_t>(k)];
                        const double dp1 = dp(0);
                        const double dp2 = dp(1);

                        // [Δt_b] = P_dt · (∂_2 p_b S_phi1
                        //                  - ∂_1 p_b S_phi2)
                        const Eigen::Matrix3d Xb =
                            dp2 * S_phi1 - dp1 * S_phi2;
                        const Eigen::Matrix3d Dt_b = P_dt * Xb;

                        Eigen::Matrix3d& Bk = Bb[static_cast<std::size_t>(k)];
                        // Flat row per Voigt index: -H_voigt[s,k]·t_0^T
                        // Curvature row per Voigt index: -φ_{0,αβ}^T·Dt_b.
                        // Row 2 is the ENGINEERING shear strain 2ρ_12 (the
                        // factor 2 matches the (ρ_11,ρ_22,2ρ_12) Voigt
                        // convention of the metric-dependent C_bend_chart,
                        // identical to the membrane B/D pairing).
                        for (int d = 0; d < 3; ++d) {
                            Bk(0, d) =
                                - H_voigt(0, k) * t0(d)
                                - (phi_11(0) * Dt_b(0, d)
                                 + phi_11(1) * Dt_b(1, d)
                                 + phi_11(2) * Dt_b(2, d));
                            Bk(1, d) =
                                - H_voigt(1, k) * t0(d)
                                - (phi_22(0) * Dt_b(0, d)
                                 + phi_22(1) * Dt_b(1, d)
                                 + phi_22(2) * Dt_b(2, d));
                            Bk(2, d) = 2.0 * (
                                - H_voigt(2, k) * t0(d)
                                - (phi_12(0) * Dt_b(0, d)
                                 + phi_12(1) * Dt_b(1, d)
                                 + phi_12(2) * Dt_b(2, d)));
                        }
                        CBb[static_cast<std::size_t>(k)] = C_chart * Bk;
                    }

                    const double scale_b = w_dA * w_A * D_bend;

                    // K_bend = scale · B_bend^T·CB is SYMMETRIC. Compute
                    // only the upper-triangle 3×3 blocks (B_a^T·CB_b) and
                    // mirror the transpose into (b, a): half the work, no
                    // dense (3·n_act)² product / ~277 KB allocation, and
                    // every product is a register-resident Matrix3d.
                    // Bit-identical to the old dense slice on the upper
                    // blocks; the lower blocks become the exact transpose
                    // (K_bend is symmetric), which only tightens symmetry.
                    for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                        const int local_a = gh.indices[
                            static_cast<std::size_t>(ia)];
                        const int a = pA.neighbor_ids[
                            static_cast<std::size_t>(local_a)];
                        const Eigen::Matrix3d BaT =
                            Bb[static_cast<std::size_t>(ia)].transpose();
                        for (Eigen::Index ib = ia; ib < n_act; ++ib) {
                            const int local_b = gh.indices[
                                static_cast<std::size_t>(ib)];
                            const int b = pA.neighbor_ids[
                                static_cast<std::size_t>(local_b)];
                            const Eigen::Matrix3d bend_block =
                                scale_b
                                * (BaT * CBb[static_cast<std::size_t>(ib)]);
                            accumulate_block(a, b, bend_block);
                            if (ib != ia) {
                                accumulate_block(b, a, bend_block.transpose());
                            }
                        }
                    }
                }

                // ----- membrane contribution -----------------------
                // Linearised Kirchhoff–Love membrane strain
                // (Millán 2011 §3.3, eq. 14):
                //   ε_αβ = ½(φ_{0,α} · u_{,β} + φ_{0,β} · u_{,α})
                // with reference-surface tangent vectors evaluated
                // from chart-basis gradients:
                //   φ_{0,α}(x_g) = Σ_b ∂_α p_b(ξ_g) · P_b
                // and inverse first fundamental form
                //   a^{αβ} = (φ_{0,α} · φ_{0,β})^{-1}_{αβ}.
                // The Kirchhoff–Love constitutive tensor is
                //   C^{αβγδ} = ν a^{αβ} a^{γδ}
                //              + ½(1-ν)(a^{αγ} a^{βδ} + a^{αδ} a^{βγ}),
                // and energy density k_L · ε^T C^{αβγδ} ε. On a flat
                // plate a^{αβ} = δ^{αβ} and the chart cardinal
                // directions are aligned with the global xy axes, so
                // this reduces to the standard plane-stress matrix
                // used by the legacy flat path.
                //
                // R10 (review 2026-06-09): this membrane term over-stiffens
                // the inextensional ovalling modes of developable shells
                // (the free-free cylinder lock — HF membrane fraction 0.444
                // vs Loop 0.039). Selective/reduced integration is the
                // classic FEM cure, but it was MEASURED not to apply here:
                // sweeping the uniform Galerkin order 12→1 moves the n=2
                // cylinder error by ~7 % against a +5300 % lock (1st-order
                // LME) — the lock is INTEGRATION-INSENSITIVE because the
                // meshfree membrane strain is non-zero for the ovalling mode
                // regardless of quadrature, so no reduced rule recovers it.
                // The lock is intrinsic to the basis, not an over-integration
                // artifact; it stays a documented Weakness (why Loop ships
                // for membrane-dominated curved shells). See
                // tests/shell/test_lme_membrane_locking_sri.cpp ([sri_probe])
                // and the [lme_cylinder_locking] note.
                if (k_mem > 0.0) {
                    // φ_{0,α} (surface tangents) and the constitutive
                    // matrix C_chart are the shared per-Gauss-point
                    // quantities computed once above (the membrane D
                    // matrix IS that same Kirchhoff–Love C, Millán §3.4).

                    // Per-node 3×3 strain-displacement blocks B_b and the
                    // constitutive product C·B_b, as FIXED-SIZE Matrix3d
                    // (so the per-pair products below stay unrolled 3×3
                    // kernels — see the bending block for the rationale).
                    std::vector<Eigen::Matrix3d>& Bm = Bm_s;
                    std::vector<Eigen::Matrix3d>& DBm = DBm_s;
                    Bm.resize(static_cast<std::size_t>(n_act));
                    DBm.resize(static_cast<std::size_t>(n_act));
                    for (Eigen::Index k = 0; k < n_act; ++k) {
                        const Eigen::VectorXd& dp = gh.gradients[
                            static_cast<std::size_t>(k)];
                        const double dp1 = dp(0);
                        const double dp2 = dp(1);
                        Eigen::Matrix3d& Bk = Bm[static_cast<std::size_t>(k)];
                        for (int i = 0; i < 3; ++i) {
                            Bk(0, i) = dp1 * phi_1(i);
                            Bk(1, i) = dp2 * phi_2(i);
                            Bk(2, i) = dp2 * phi_1(i) + dp1 * phi_2(i);
                        }
                        DBm[static_cast<std::size_t>(k)] = C_chart * Bk;
                    }
                    const double scale_m = w_dA * w_A * k_mem;

                    // K_mem = scale · B_mem^T·DB is SYMMETRIC — same
                    // upper-triangle-only treatment as the bending block.
                    for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                        const int local_a = gh.indices[
                            static_cast<std::size_t>(ia)];
                        const int a = pA.neighbor_ids[
                            static_cast<std::size_t>(local_a)];
                        const Eigen::Matrix3d BaT =
                            Bm[static_cast<std::size_t>(ia)].transpose();
                        for (Eigen::Index ib = ia; ib < n_act; ++ib) {
                            const int local_b = gh.indices[
                                static_cast<std::size_t>(ib)];
                            const int b = pA.neighbor_ids[
                                static_cast<std::size_t>(local_b)];
                            const Eigen::Matrix3d mem_block =
                                scale_m
                                * (BaT * DBm[static_cast<std::size_t>(ib)]);
                            accumulate_block(a, b, mem_block);
                            if (ib != ia) {
                                accumulate_block(b, a, mem_block.transpose());
                            }
                        }
                    }
                }
            }

            // Renormalise the surviving Shepard weights when any
            // patch was dropped at this Gauss point: a successful
            // patch's contribution was accumulated with the
            // original w_A but the post-drop effective weight is
            // w_A / (1 - w_failed_sum) to preserve partition of
            // unity. Hoists the per-(a, b) divide out of the inner
            // loop by scaling the gauss_blocks once.
            if (w_failed_sum > 0.0) {
                if (!(w_failed_sum < 1.0)) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_K (curved): every "
                        "Shepard-active patch's in-chart Newton "
                        "diverged at a quadrature point — the local "
                        "chart neighbourhoods are entirely "
                        "incompatible with this Gauss point.");
                }
                // Silent-degradation guard: a Gauss point may shed routine
                // ownership-filter weight, but if the in-chart-Newton drop
                // net sheds more than the configured fraction of this point's
                // PoU, the renormalised block is too perturbed to trust —
                // surface it instead of folding it in silently.
                if (w_newton_sum > params.max_newton_drop_frac) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_K (curved): in-chart-Newton "
                        "drop net shed " + std::to_string(w_newton_sum)
                        + " of a quadrature point's partition of unity "
                        "(> max_newton_drop_frac "
                        + std::to_string(params.max_newton_drop_frac)
                        + ") — local chart breakdown, not a benign "
                        "renormalisation. Remesh or raise the threshold.");
                }
                // Total-PoU-loss guard (review3 N2): the ownership-filter
                // channel (w_failed_sum − w_newton_sum) is excluded from the
                // Newton budget above, yet it drives this same renormalisation
                // denominator. Cap the COMBINED loss so the rescale below can
                // never silently absorb a near-total PoU shed.
                if (w_failed_sum > params.max_total_drop_frac) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_K (curved): drop net shed "
                        + std::to_string(w_failed_sum)
                        + " of a quadrature point's partition of unity "
                        "(> max_total_drop_frac "
                        + std::to_string(params.max_total_drop_frac)
                        + ") — ownership-filter + Newton drops left too "
                        "little surviving weight to renormalise reliably. "
                        "Remesh or raise the threshold.");
                }
                {
                    auto& dt = drops_per_thread[
                        static_cast<std::size_t>(tid)];
                    dt.w_gauss_max =
                        std::max(dt.w_gauss_max, w_failed_sum);
                    if (w_failed_sum > 0.01) ++dt.n_gauss_gt001;
                    if (w_failed_sum > 0.1)  ++dt.n_gauss_gt01;
                    if (w_failed_sum > 0.3)  ++dt.n_gauss_gt03;
                }
                const double rescale = 1.0 / (1.0 - w_failed_sum);
                for (auto& [key, block] : gauss_blocks) {
                    block *= rescale;
                }
            }

            // Fold this Gauss point's blocks into the per-cell
            // accumulator.
            for (const auto& [key, block] : gauss_blocks) {
                auto it = thread_blocks.find(key);
                if (it == thread_blocks.end()) {
                    thread_blocks.emplace(key, block);
                } else {
                    it->second += block;
                }
            }
        }

        }  // for t

        // Emit one triplet per unique (a, b) accumulated across this thread's
        // ENTIRE triangle range (folded once, not re-emitted per triangle).
        // Decode using the same n_ext divisor that block_key uses above —
        // a/b are extended-space indices in [0, n_ext) when use_ghost_nodes
        // is on. setFromTriplets later sums any cross-thread (a, b) splits.
        for (const auto& [key, block] : thread_blocks) {
            const int a = static_cast<int>(key / n_ext);
            const int b = static_cast<int>(key % n_ext);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double v = block(i, j);
                    if (v != 0.0) {
                        trips.emplace_back(3 * a + i, 3 * b + j, v);
                    }
                }
            }
        }
    };  // worker

    // Spawn n_threads-1 workers + run the last shard on this thread;
    // run_threaded rethrows any worker exception here (e.g. an
    // infeasible-r_cut Newton divergence) rather than std::terminate-ing.
    run_threaded(n_threads, worker);

    // Reduce the per-thread tallies unconditionally — the basic
    // counters feed LMEAssembler::last_drop_stats() (the queryable B1
    // watchdog); only the verbose breakdown below stays env-gated.
    DropTally tot;
    std::unordered_map<int, long> by_anchor;
    for (const auto& dt : drops_per_thread) {
        tot.n_own       += dt.n_own;
        tot.n_newton    += dt.n_newton;
        tot.n_escal_ok  += dt.n_escal_ok;
        tot.escal_steps += dt.escal_steps;
        tot.w_own       += dt.w_own;
        tot.w_newton    += dt.w_newton;
        tot.w_gauss_max  = std::max(tot.w_gauss_max, dt.w_gauss_max);
        tot.n_gauss_gt001 += dt.n_gauss_gt001;
        tot.n_gauss_gt01  += dt.n_gauss_gt01;
        tot.n_gauss_gt03  += dt.n_gauss_gt03;
        for (const auto& [a, c] : dt.newton_by_anchor)
            by_anchor[a] += c;
    }
    std::unordered_map<int, long> escal_anchor;
    for (const auto& dt : drops_per_thread)
        for (const auto& [a, c] : dt.escal_by_anchor)
            escal_anchor[a] += c;
    if (drop_stats_out != nullptr) {
        drop_stats_out->n_own       = tot.n_own;
        drop_stats_out->n_newton    = tot.n_newton;
        drop_stats_out->n_escal_ok  = tot.n_escal_ok;
        drop_stats_out->w_own       = tot.w_own;
        drop_stats_out->w_newton    = tot.w_newton;
        drop_stats_out->w_gauss_max = tot.w_gauss_max;
    }
    if (diag_drops) {
        std::fprintf(stderr,
            "[lme_diag K] drops: ownership n=%ld (w=%.3g)  "
            "newton n=%ld (w=%.3g)  anchors_with_newton_fail=%zu  "
            "alpha_escal rescued=%ld (steps=%ld)\n"
            "[lme_diag K] per-gauss dropped weight: max=%.3g  "
            "#>0.01=%ld  #>0.1=%ld  #>0.3=%ld\n",
            tot.n_own, tot.w_own, tot.n_newton, tot.w_newton,
            by_anchor.size(), tot.n_escal_ok, tot.escal_steps,
            tot.w_gauss_max, tot.n_gauss_gt001, tot.n_gauss_gt01,
            tot.n_gauss_gt03);
        std::vector<std::pair<int, long>> ranked(
            by_anchor.begin(), by_anchor.end());
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& x, const auto& y) {
                      return x.second > y.second;
                  });
        const std::size_t n_show = std::min<std::size_t>(ranked.size(), 16);
        for (std::size_t i = 0; i < n_show; ++i) {
            const int a = ranked[i].first;
            std::fprintf(stderr,
                "  [lme_diag K] anchor %d  fails=%ld  |x|=%.5g  "
                "chart_n=%zu\n",
                a, ranked[i].second, V_ext.row(a).norm(),
                patches[static_cast<std::size_t>(a)].neighbor_ids.size());
        }
        // Where does the B3 α-escalation ladder fire? Anchor positions
        // localise the rescued Gauss points (e.g. pinch corners).
        std::vector<std::pair<int, long>> eranked(
            escal_anchor.begin(), escal_anchor.end());
        std::sort(eranked.begin(), eranked.end(),
                  [](const auto& x, const auto& y) {
                      return x.second > y.second;
                  });
        const std::size_t n_eshow =
            std::min<std::size_t>(eranked.size(), 16);
        for (std::size_t i = 0; i < n_eshow; ++i) {
            const int a = eranked[i].first;
            std::fprintf(stderr,
                "  [lme_diag K] escal anchor %d  rescues=%ld  "
                "x=(%.4g %.4g %.4g)\n",
                a, eranked[i].second,
                V_ext(a, 0), V_ext(a, 1), V_ext(a, 2));
        }
    }

    // Concatenate per-thread triplets. setFromTriplets handles
    // duplicates correctly so no cross-thread (a, b) dedup is needed.
    std::size_t total = 0;
    for (const auto& tt : trips_per_thread) total += tt.size();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(total);
    for (auto& tt : trips_per_thread) {
        trips.insert(trips.end(),
            std::make_move_iterator(tt.begin()),
            std::make_move_iterator(tt.end()));
    }

    Eigen::SparseMatrix<double> K(3 * n_ext, 3 * n_ext);
    K.setFromTriplets(trips.begin(), trips.end());
    K.makeCompressed();
    // Belt-and-suspenders finiteness check on the assembled stream: the
    // per-Gauss-point metric guard above is the root-cause fix for the
    // degenerate-chart NaN, but a single O(nnz) sweep here catches any
    // other non-finite triplet source (e.g. a NaN leaking in through
    // V_ext) before it becomes an opaque downstream eigensolver failure.
    if (!K.coeffs().allFinite()) {
        throw std::runtime_error(
            "LMEAssembler::assemble_K (curved): assembled stiffness "
            "matrix contains a non-finite entry — a degenerate chart or "
            "non-finite input geometry corrupted K. Remesh / check the "
            "input vertex positions.");
    }
    return K;
}

/// Curved-shell @ref LMEAssembler::assemble_M.
///
/// Mirrors the structure of @ref assemble_K_curved_bending but
/// integrates the consistent mass bilinear form
/// @f$ M_{ab} = \int_{\mathcal M} \rho h\, p_a(x)\, p_b(x)\,d\mathcal M @f$
/// on the input F triangles in R³ (Strategy P2 quadrature) with
/// Shepard-PoU blending across the per-vertex tangent charts. The
/// three displacement components decouple at the mass level
/// (Kirchhoff plate decoupling), so each (a, b) contribution is a
/// scalar M_ab times the 3×3 identity; per-Gauss-point block dedup
/// keeps global triplet emissions bounded.
[[nodiscard]] Eigen::SparseMatrix<double> assemble_M_curved(
    const Eigen::MatrixXd&                       V,
    const Eigen::MatrixXi&                       F,
    double                                       surface_density,
    const LMEAssembler::Params&                  params)
{
    if (V.cols() != 3) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_M (curved): V must have 3 columns");
    }

    const Eigen::Index n_v = V.rows();

    // Shared node-set + chart construction — identical to
    // assemble_K_curved_bending so K and M live on the same discrete
    // function space (the mass path evaluates the basis per patch via
    // pA.xi, so it needs no V_ext alias). Aliased so the code downstream
    // reads as before.
    const CurvedPatchContext _ctx = build_curved_patch_context(V, F, params);
    const auto&        bdry         = _ctx.bdry;
    const auto&        h_a_ext      = _ctx.h_a_ext;
    const Eigen::Index n_ext        = _ctx.n_ext;
    const auto&        beta_lme     = _ctx.beta_lme;
    const auto&        beta_patches = _ctx.beta_patches;
    const double       r_cut        = _ctx.r_cut;
    const auto&        adjacency    = _ctx.adjacency;
    const auto&        patches      = _ctx.patches;
    const auto&        patch_points = _ctx.patch_points;

    // Per-patch neighbour membership for the faithful per-patch quadrature
    // ownership test — see the matching comment in assemble_K_curved_bending.
    std::vector<std::unordered_set<int>> patch_nbr_set(
        static_cast<std::size_t>(patches.size()));
    for (std::size_t A = 0; A < patches.size(); ++A) {
        const auto& nids = patches[A].neighbor_ids;
        patch_nbr_set[A].reserve(nids.size() * 2);
        patch_nbr_set[A].insert(nids.begin(), nids.end());
    }

    // SME INTERIOR ownership — mirrors assemble_K_curved_bending; see
    // the full rationale there (rim-infeasibility root cause,
    // 2026-06-07): every SME quadrature query must keep a full 1-ring
    // of chart-node surround for the moment-with-gap program to be
    // feasible.
    std::vector<std::unordered_set<int>> patch_interior_set;
    if (params.use_second_order_sme) {
        patch_interior_set.resize(patches.size());
        for (std::size_t A = 0; A < patches.size(); ++A) {
            const auto& nbr = patch_nbr_set[A];
            auto&       out = patch_interior_set[A];
            out.reserve(nbr.size());
            for (int v : patches[A].neighbor_ids) {
                if (v < 0
                    || v >= static_cast<int>(adjacency.size()))
                    continue;
                bool interior = true;
                for (int u :
                     adjacency[static_cast<std::size_t>(v)]) {
                    if (!nbr.count(u)) { interior = false; break; }
                }
                if (interior) out.insert(v);
            }
        }
    }

    // ----- 2nd-order SME nodal-gap matrices (paper §3.2) ------------
    // Mirrors assemble_K_curved_bending's setup — see the helper
    // docstring for the classification.
    std::vector<std::vector<Eigen::Matrix2d>> sme_gaps_per_patch;
    if (params.use_second_order_sme) {
        std::vector<lme::BoundaryEdge> bdry_owned;
        const std::vector<lme::BoundaryEdge>* bdry_for_sme = &bdry;
        if (!params.use_ghost_nodes) {
            bdry_owned   = lme::collect_boundary_edges(F);
            bdry_for_sme = &bdry_owned;
        }
        const std::vector<Eigen::Matrix3d> node_cov =
            vertex_one_ring_covariance(V, F);
        const std::vector<std::vector<Eigen::Vector3d>> one_ring_edges =
            vertex_one_ring_edges(V, F);
        const std::vector<VtxBoundaryFrame> sme_frames =
            compute_sme_boundary_frames(
                V, *bdry_for_sme, adjacency, n_v);
        sme_gaps_per_patch = build_per_patch_sme_gaps(
            patches, sme_frames, n_v, h_a_ext, node_cov, one_ring_edges,
            params.sme_alpha, params.sme_beta);
    }

    const auto qpts = chladni::shell::quadrature_points(
        quadrature_rule_from_count(params.n_quadrature_per_tri));

    // Parallel triangle loop — same shape as assemble_K_curved_bending.
    const unsigned hw_m = std::thread::hardware_concurrency();
    const int n_threads_raw = static_cast<int>(hw_m == 0 ? 1u : hw_m);
    const Eigen::Index n_F = F.rows();
    const int n_threads = static_cast<int>(
        std::min<Eigen::Index>(n_threads_raw, std::max<Eigen::Index>(1, n_F)));
    std::vector<std::vector<Eigen::Triplet<double>>> trips_per_thread(
        static_cast<std::size_t>(n_threads));

    auto block_key = [n_ext_l = static_cast<std::int64_t>(n_ext)](
                          int a, int b) -> std::int64_t {
        return static_cast<std::int64_t>(a) * n_ext_l
             + static_cast<std::int64_t>(b);
    };

    auto worker = [&](int tid) {
        std::unordered_map<std::int64_t, double> gauss_scalar;
        // Per-THREAD (a, b) accumulator — folds every triangle's contribution
        // before emitting, so the triplet stream is O(nnz) not
        // O(nnz × support-triangles). See the matching note in
        // assemble_K_curved_bending.
        std::unordered_map<std::int64_t, double> thread_scalar;
        gauss_scalar.reserve(512);
        thread_scalar.reserve(4096);
        auto& trips = trips_per_thread[static_cast<std::size_t>(tid)];
        trips.reserve(
            static_cast<std::size_t>(3 * n_ext) * 3 * 32
            / static_cast<std::size_t>(n_threads));

        auto accumulate_scalar = [&](int a, int b, double v) {
            const std::int64_t key = block_key(a, b);
            auto it = gauss_scalar.find(key);
            if (it == gauss_scalar.end()) {
                gauss_scalar.emplace(key, v);
            } else {
                it->second += v;
            }
        };

        const Eigen::Index chunk = (n_F + n_threads - 1) / n_threads;
        const Eigen::Index t_start = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index t_end   = std::min(t_start + chunk, n_F);

        for (Eigen::Index t = t_start; t < t_end; ++t) {
            // NB: thread_scalar is NOT cleared per triangle — it accumulates
            // across this thread's whole range; emitted once after the loop.
        const int i0 = F(t, 0), i1 = F(t, 1), i2 = F(t, 2);
        const Eigen::Vector3d P0 = V.row(i0).transpose();
        const Eigen::Vector3d P1 = V.row(i1).transpose();
        const Eigen::Vector3d P2 = V.row(i2).transpose();
        const Eigen::Vector3d e01 = P1 - P0;
        const Eigen::Vector3d e02 = P2 - P0;
        const double          two_area = e01.cross(e02).norm();

        for (const auto& q : qpts) {
            const double u_bary = 1.0 - q.v - q.w;
            const Eigen::Vector3d Pq =
                u_bary * P0 + q.v * P1 + q.w * P2;

            const double w_pq = q.weight * two_area * surface_density;

            // PoU truncation = kPoUTruncTol (shared with the stiffness path
            // so K and M live on the SAME function space).
            const double tol_pu = kPoUTruncTol;
            const lme::ShepardWeights sw = lme::shepard_partition(
                patch_points, beta_patches, Pq, tol_pu);

            gauss_scalar.clear();
            // Matches the catch-and-renormalise safety net in
            // assemble_K_curved_bending; see the comment there.
            double w_failed_sum = 0.0;
            double w_newton_sum = 0.0;  // Newton-only; abort budget

            for (std::size_t s = 0; s < sw.indices.size(); ++s) {
                const int                A   = sw.indices[s];
                const double             w_A = sw.values[s];
                const lme::Patch&        pA  = patches[
                    static_cast<std::size_t>(A)];

                // Faithful per-patch quadrature ownership test — see the
                // matching comment in assemble_K_curved_bending.
                const auto& nbrA =
                    params.use_second_order_sme
                        ? patch_interior_set[static_cast<std::size_t>(A)]
                        : patch_nbr_set[static_cast<std::size_t>(A)];
                if (!nbrA.count(i0) || !nbrA.count(i1) || !nbrA.count(i2)) {
                    w_failed_sum += w_A;
                    continue;
                }

                // Chart projection of the Gauss point.
                const Eigen::Vector3d delta_g = Pq - pA.Qbar;
                Eigen::Vector2d       xi_g    = pA.V.transpose() * delta_g;

                // Per-neighbour β slice for the in-chart 2D solve.
                const Eigen::Index n_neigh =
                    static_cast<Eigen::Index>(pA.neighbor_ids.size());
                Eigen::VectorXd beta_chart(n_neigh);
                for (Eigen::Index k = 0; k < n_neigh; ++k) {
                    beta_chart(k) = beta_lme(
                        pA.neighbor_ids[static_cast<std::size_t>(k)]);
                }

                // Basis values only — gradients/Hessians not needed
                // for mass assembly. Catch in-chart Newton
                // divergence and drop the patch; the renormalise
                // step below preserves partition of unity. The
                // catch is also required for thread safety; see
                // assemble_K_curved_bending for the rationale.
                LMEBasisValues bv;
                try {
                    if (params.use_second_order_sme) {
                        // Renormalise chart to h_chart=1; see the
                        // assemble_K_curved_bending dispatch for the
                        // rationale (SME Newton scale-sensitivity).
                        // Basis values are dimensionless so no
                        // back-scaling on the output is needed.
                        const int sme_iters_cap = std::max(
                            params.newton_max_iters,
                            5 * params.newton_max_iters / 2);
                        const double h_chart =
                            h_a_ext(pA.anchor_id);
                        const double inv_h = 1.0 / h_chart;
                        const Eigen::MatrixXd xi_scaled =
                            pA.xi * inv_h;
                        const Eigen::Vector2d xi_g_scaled =
                            xi_g * inv_h;
                        std::vector<Eigen::Matrix2d> d_scaled =
                            sme_gaps_per_patch[
                                static_cast<std::size_t>(A)];
                        const double inv_h2 = inv_h * inv_h;
                        for (auto& dk : d_scaled) dk *= inv_h2;
                        // Per-patch α escalation — mirrors
                        // assemble_K_curved_bending (gap matrices
                        // enlarged ×2 per retry before the drop-net;
                        // see the comment there).
                        for (int esc = 0;; ++esc) {
                            try {
                                bv = lme::evaluate_sme_basis(
                                    xi_scaled, d_scaled,
                                    xi_g_scaled, r_cut * inv_h,
                                    params.newton_tol, sme_iters_cap);
                                break;
                            } catch (const std::runtime_error&) {
                                if (esc >= std::max(
                                        0,
                                        params
                                            .sme_alpha_escalation_steps))
                                    throw;
                                for (auto& dk : d_scaled) dk *= 2.0;
                            }
                        }
                    } else {
                        bv = lme::evaluate_basis(
                            pA.xi, beta_chart, xi_g, r_cut,
                            params.newton_tol, params.newton_max_iters);
                    }
                } catch (const std::runtime_error&) {
                    w_failed_sum += w_A;
                    w_newton_sum += w_A;  // counts toward the abort budget
                    continue;
                }

                const Eigen::Index n_act =
                    static_cast<Eigen::Index>(bv.indices.size());
                const double       scale = w_pq * w_A;

                for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                    const int    lid_a = bv.indices[
                        static_cast<std::size_t>(ia)];
                    const int    a   = pA.neighbor_ids[
                        static_cast<std::size_t>(lid_a)];
                    const double p_a = bv.values[
                        static_cast<std::size_t>(ia)];
                    if (p_a == 0.0) continue;
                    for (Eigen::Index ib = 0; ib < n_act; ++ib) {
                        const int    lid_b = bv.indices[
                            static_cast<std::size_t>(ib)];
                        const int    b   = pA.neighbor_ids[
                            static_cast<std::size_t>(lid_b)];
                        const double p_b = bv.values[
                            static_cast<std::size_t>(ib)];
                        if (p_b == 0.0) continue;
                        accumulate_scalar(a, b, scale * p_a * p_b);
                    }
                }
            }

            // Renormalise on failure; see assemble_K_curved_bending
            // for the rationale.
            if (w_failed_sum > 0.0) {
                if (!(w_failed_sum < 1.0)) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_M (curved): every "
                        "Shepard-active patch's in-chart Newton "
                        "diverged at a quadrature point.");
                }
                if (w_newton_sum > params.max_newton_drop_frac) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_M (curved): in-chart-Newton "
                        "drop net shed " + std::to_string(w_newton_sum)
                        + " of a quadrature point's partition of unity "
                        "(> max_newton_drop_frac "
                        + std::to_string(params.max_newton_drop_frac)
                        + ") — local chart breakdown.");
                }
                // Total-PoU-loss guard (review3 N2): cap the COMBINED
                // ownership-filter + Newton loss driving this denominator.
                if (w_failed_sum > params.max_total_drop_frac) {
                    throw std::runtime_error(
                        "LMEAssembler::assemble_M (curved): drop net shed "
                        + std::to_string(w_failed_sum)
                        + " of a quadrature point's partition of unity "
                        "(> max_total_drop_frac "
                        + std::to_string(params.max_total_drop_frac)
                        + ") — too little surviving weight to renormalise "
                        "reliably.");
                }
                const double rescale = 1.0 / (1.0 - w_failed_sum);
                for (auto& [key, val] : gauss_scalar) {
                    val *= rescale;
                }
            }

            // Fold this Gauss point's scalar accumulator into the
            // per-cell accumulator.
            for (const auto& [key, val] : gauss_scalar) {
                auto it = thread_scalar.find(key);
                if (it == thread_scalar.end()) {
                    thread_scalar.emplace(key, val);
                } else {
                    it->second += val;
                }
            }
        }

        }  // for t

        // Emit 3 diagonal entries per unique (a, b) accumulated across this
        // thread's ENTIRE range (folded once, not per triangle). Decode using
        // the same n_ext divisor that block_key uses; setFromTriplets sums any
        // cross-thread (a, b) splits.
        for (const auto& [key, val] : thread_scalar) {
            if (val == 0.0) continue;
            const int a = static_cast<int>(key / n_ext);
            const int b = static_cast<int>(key % n_ext);
            for (int d = 0; d < 3; ++d) {
                trips.emplace_back(3 * a + d, 3 * b + d, val);
            }
        }
    };  // worker

    run_threaded(n_threads, worker);

    std::size_t total = 0;
    for (const auto& tt : trips_per_thread) total += tt.size();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(total);
    for (auto& tt : trips_per_thread) {
        trips.insert(trips.end(),
            std::make_move_iterator(tt.begin()),
            std::make_move_iterator(tt.end()));
    }

    Eigen::SparseMatrix<double> M(3 * n_ext, 3 * n_ext);
    M.setFromTriplets(trips.begin(), trips.end());
    M.makeCompressed();
    // Belt-and-suspenders finiteness check, mirroring assemble_K_curved
    // above: the triplet emitter filters val == 0.0, which a NaN passes
    // (NaN != 0), so a non-finite basis value or vertex position would
    // otherwise reach the generalized eigenproblem as an opaque failure.
    // M never inverts the metric, so this O(nnz) sweep is its only guard.
    if (!M.coeffs().allFinite()) {
        throw std::runtime_error(
            "LMEAssembler::assemble_M (curved): assembled mass "
            "matrix contains a non-finite entry — a degenerate chart or "
            "non-finite input geometry corrupted M. Remesh / check the "
            "input vertex positions.");
    }
    return M;
}

}  // namespace

LMEAssembler::LMEAssembler(Params params) : params_(params)
{
    // The drop-fraction budgets are upper bounds on a PoU loss in [0, 1).
    // A negative value would make the guard throw on ANY drop (w_failed_sum >
    // negative once w_failed_sum > 0) — a confusing self-inflicted abort.
    // The ">= 1.0 disables it" contract is honoured for the upper end; pin
    // the lower end so a typo can't silently invert the guard's meaning.
    if (!(params_.max_newton_drop_frac >= 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: max_newton_drop_frac must be >= 0 "
            "(>= 1.0 disables the abort).");
    }
    if (!(params_.max_total_drop_frac >= 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: max_total_drop_frac must be >= 0 "
            "(>= 1.0 disables the abort).");
    }

    // Chart-sizing parameters feed unguarded sqrt(-log(·)) expressions in
    // build_curved_patch_context — c.r_cut = sqrt(-log(tol_lme)/gamma)·h and
    // lme_chart_factor = sqrt(-log(chart_tol_lme)/gamma) — which run BEFORE
    // any lme::evaluate_basis call, so the per-call boundary checks the basis
    // evaluator performs do not cover them. A bad value here produces a NaN
    // radius that silently collapses the chart to the 2-ring fallback, or a
    // late, opaque "no nodes within r_cut" throw from inside Newton, or an
    // uncaught invalid_argument escaping run_threaded. Reject up front. The
    // `!(x op)` form also rejects NaN. These mirror the require_positive
    // helper in compute_nodal_gaps.
    if (!(params_.gamma > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: gamma must be > 0 (it divides the chart radius "
            "sqrt(-log(tol_lme)/gamma)).");
    }
    if (!(params_.tol_lme > 0.0 && params_.tol_lme < 1.0)) {
        throw std::invalid_argument(
            "LMEAssembler: tol_lme must lie in (0, 1) so -log(tol_lme) > 0 "
            "(it sets the basis-support cutoff radius).");
    }
    if (!(params_.chart_tol_lme > 0.0 && params_.chart_tol_lme < 1.0)) {
        throw std::invalid_argument(
            "LMEAssembler: chart_tol_lme must lie in (0, 1) so "
            "-log(chart_tol_lme) > 0 (it sets the LME chart factor).");
    }
    if (!(params_.r_cut_mult > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: r_cut_mult must be > 0 (it scales the flat-path "
            "truncation radius r_cut_mult·h).");
    }
    // gamma_wpca and gamma_pu are the β-rate twins of gamma: they feed the
    // identical gamma_x / h² form (beta_wpca, beta_patches) in
    // build_curved_patch_context, again before any evaluate_basis check. A
    // non-positive value throws late and opaquely from build_patch /
    // shepard_partition, attributed to a per-node/per-patch index rather than
    // to the knob the user set (review7 I3).
    if (!(params_.gamma_wpca > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: gamma_wpca must be > 0 (it sets the wPCA metric "
            "rate beta_wpca = gamma_wpca / h²).");
    }
    if (!(params_.gamma_pu > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: gamma_pu must be > 0 (it sets the Shepard "
            "partition-of-unity rate beta_patches = gamma_pu / h²).");
    }
    // max_chart_nodes caps the per-anchor chart in chart_bfs_core. A
    // non-positive value forms `sel.begin() + max_nodes` (out-of-range iterator)
    // and `sel.resize(static_cast<size_t>(max_nodes))` (negative → ~1.8e19),
    // i.e. UB / bad_alloc in the chart build, before any evaluate_basis check.
    // 7 is the hardcoded min_nodes floor, so the cap must never undercut it (J3).
    if (!(params_.max_chart_nodes >= 7)) {
        throw std::invalid_argument(
            "LMEAssembler: max_chart_nodes must be >= 7 (it caps the per-anchor "
            "chart; below the min-node floor the chart build is undefined).");
    }
    // Per-Gauss-point Newton controls. newton_tol <= 0 (or NaN) makes the
    // convergence test unreachable so Newton always exhausts its budget and
    // throws "did not converge"; newton_max_iters < 1 throws immediately — both
    // from a worker thread, misattributed to the mesh rather than the knob (J4,
    // cf. I1/I2).
    if (!(params_.newton_tol > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler: newton_tol must be > 0 (the LME/SME dual convergence "
            "tolerance; <= 0 can never be met).");
    }
    if (params_.newton_max_iters < 1) {
        throw std::invalid_argument(
            "LMEAssembler: newton_max_iters must be >= 1.");
    }
    // n_quadrature_per_tri must be one of {1,3,7,12}. quadrature_rule_from_count
    // already enforces this, but only at assembly time after the (expensive)
    // chart build; call it once here to fail fast on a config typo (J5).
    (void)quadrature_rule_from_count(params_.n_quadrature_per_tri);
    // sme_alpha / sme_beta are consulted ONLY when use_second_order_sme is on
    // (they parameterise compute_nodal_gaps and the SME cutoff radius). Their
    // governing consumer requires alpha > 1 and beta >= 1 — STRICTER than the
    // > 0 the radius alone needs — so validate to the consumer's bound, not
    // the radius'. Without this an alpha in (0, 1] (or beta < 1) passes the
    // ctor and throws late from compute_nodal_gaps on a worker thread, the
    // exact late/opaque failure this up-front gate exists to prevent
    // (review7 I1/I2).
    if (params_.use_second_order_sme) {
        if (!(params_.sme_alpha > 1.0)) {
            throw std::invalid_argument(
                "LMEAssembler: sme_alpha must be > 1 when use_second_order_sme "
                "(the SME nodal-gap recipe requires alpha > 1; the paper "
                "operates in [1.6, 2.5]).");
        }
        if (!(params_.sme_beta >= 1.0)) {
            throw std::invalid_argument(
                "LMEAssembler: sme_beta must be >= 1 when use_second_order_sme "
                "(compute_nodal_gaps requires beta >= 1).");
        }
    }
}

Eigen::SparseMatrix<double> LMEAssembler::assemble_K(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ShellMaterial&   material) const
{
    // ----- preconditions ---------------------------------------------
    if (V.rows() == 0 || F.rows() == 0) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_K: V and F must be non-empty");
    }
    // The flat-plate Kirchhoff--Love decomposition gives two decoupled
    // contributions to K: bending on the z displacement (driven by
    // material.k_B) and 2D plane-stress membrane on the in-plane (x, y)
    // displacements (driven by material.k_L). At least one must be
    // positive — with both zero there is nothing to assemble.
    if (!(material.k_B > 0.0) && !(material.k_L > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_K: at least one of material.k_L "
            "(membrane) or material.k_B (bending) must be > 0 (got k_L="
            + std::to_string(material.k_L) + ", k_B="
            + std::to_string(material.k_B) + ")");
    }
    if (params_.use_second_order_sme && !params_.use_curved_shell) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_K: use_second_order_sme requires "
            "use_curved_shell=true (the SME basis evaluates on the "
            "per-patch wPCA chart, which only exists on the curved path)");
    }

    // Curved-shell path (Millan 2011 wPCA charts + Shepard PoU):
    // when the caller opts into the new formulation, dispatch to the
    // patch-based curved assembly. Both contributions are implemented —
    // Kirchhoff–Love bending (driven by k_B) and the curved-metric
    // membrane (driven by k_L), assembled from the shared C_chart built
    // from the inverse first fundamental form — so the only material
    // requirement is the at-least-one-of-k_L/k_B guard above; a
    // membrane-only (k_B == 0) or bending-only (k_L == 0) material is
    // accepted. The curved assembly accepts both flat and curved meshes
    // and reduces algebraically to the flat path on planar input.
    if (params_.use_curved_shell) {
        return assemble_K_curved_bending(V, F, material, params_,
                                         &last_drops_);
    }

    require_flat_plate(V);

    // ----- geometry / locality ---------------------------------------
    const Eigen::Index n_v = V.rows();
    Eigen::MatrixXd nodes2d = V.leftCols(2);

    const Eigen::VectorXd h_a = vertex_one_ring_h(V, F);
    Eigen::VectorXd beta(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        beta(a) = params_.gamma / (h_a(a) * h_a(a));
    }
    const double r_cut = params_.r_cut_mult * h_a.maxCoeff();

    const auto qpts =
        chladni::shell::quadrature_points(QuadratureRule::SevenPointDunavant);

    // ----- bending compliance matrix --------------------------------
    // Kirchhoff plate bending bilinear form:
    //   a_B(u, v) = D ∫ [Δu·Δv − (1−ν)(u_xx v_yy + u_yy v_xx − 2 u_xy v_xy)]
    // expands (in the (H_xx, H_yy, H_xy) basis) as
    //   integrand = D · h_a^T · C_bend(ν) · h_b
    // with C_bend symmetric, positive-semidefinite for ν ∈ [0, 1]:
    //   C_bend = [[1, ν, 0], [ν, 1, 0], [0, 0, 2(1−ν)]].
    // Eigenvalues 1+ν, 1−ν, 2(1−ν) — all nonneg in that range.
    const double D_bend  = material.k_B;          // bending modulus (J)
    const double k_mem   = material.k_L;          // membrane modulus (N/m)
    const double nu      = material.poisson_ratio;
    Eigen::Matrix3d C_bend;
    C_bend << 1.0, nu,  0.0,
              nu,  1.0, 0.0,
              0.0, 0.0, 2.0 * (1.0 - nu);

    // ----- 2D plane-stress membrane compliance ----------------------
    // Standard Kirchhoff--Love membrane on a flat plate (Belytschko
    // 1994; Reissner-Mindlin reduction at vanishing curvature):
    //   a_M(u, v) = k_L ∫ ε(u)^T · D_mem · ε(v)
    // with engineering strain Voigt vector
    //   ε = (∂u_x/∂x, ∂u_y/∂y, ∂u_x/∂y + ∂u_y/∂x)
    // and plane-stress compliance
    //   D_mem = [[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]].
    // The (1−ν)/2 factor on the shear row follows from using the
    // engineering shear γ_xy = 2 ε_xy in the Voigt convention. The
    // bending and membrane bilinear forms decouple on a flat plate
    // (no membrane-bending coupling at zero curvature), so K below is
    // the direct sum of the two.
    Eigen::Matrix3d D_mem;
    D_mem << 1.0, nu,  0.0,
             nu,  1.0, 0.0,
             0.0, 0.0, 0.5 * (1.0 - nu);

    // ----- triplets for the two scalar / block decompositions --------
    // Per-cell accumulators keyed on (a, b): all Gauss points within a
    // single triangle contribute to the same (a, b) entries, so we
    // sum them in dense per-cell maps and emit ONE triplet per (a, b)
    // pair per triangle rather than per (Gauss point × pair). For a
    // 7-point Dunavant rule this trims triplet emission by 7× and
    // shrinks the final setFromTriplets sort/merge accordingly.
    //
    // The bend accumulator is a scalar per pair; the membrane
    // accumulator stores a 2x2 block (the engineering-shear
    // components couple in-plane DOFs).
    // Parallel triangle loop — same shape as the curved paths.
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_threads_raw = static_cast<int>(hw == 0 ? 1u : hw);
    const Eigen::Index n_F = F.rows();
    const int n_threads = static_cast<int>(
        std::min<Eigen::Index>(n_threads_raw, std::max<Eigen::Index>(1, n_F)));
    std::vector<std::vector<Eigen::Triplet<double>>> bend_per_thread(
        static_cast<std::size_t>(n_threads));
    std::vector<std::vector<Eigen::Triplet<double>>> mem_per_thread(
        static_cast<std::size_t>(n_threads));

    auto pair_key = [n_v_l = static_cast<std::int64_t>(n_v)](
                        int a, int b) -> std::int64_t {
        return static_cast<std::int64_t>(a) * n_v_l
             + static_cast<std::int64_t>(b);
    };

    auto worker = [&](int tid) {
        std::unordered_map<std::int64_t, double>          bend_acc;
        std::unordered_map<std::int64_t, Eigen::Matrix2d> mem_acc;
        // Per-THREAD (a, b) accumulators — folded across all this thread's
        // triangles before emitting (O(nnz), not O(nnz × support-triangles)).
        // See assemble_K_curved_bending.
        bend_acc.reserve(4096);
        mem_acc.reserve(4096);
        auto& trips_bend = bend_per_thread[static_cast<std::size_t>(tid)];
        auto& trips_mem  = mem_per_thread[static_cast<std::size_t>(tid)];
        if (D_bend > 0.0) {
            trips_bend.reserve(static_cast<std::size_t>(n_F) * 144
                               / static_cast<std::size_t>(n_threads));
        }
        if (k_mem > 0.0) {
            trips_mem.reserve(static_cast<std::size_t>(n_F) * 4 * 144
                              / static_cast<std::size_t>(n_threads));
        }

        const Eigen::Index chunk = (n_F + n_threads - 1) / n_threads;
        const Eigen::Index t_start = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index t_end   = std::min(t_start + chunk, n_F);

        Eigen::VectorXd x_q(2);
        for (Eigen::Index t = t_start; t < t_end; ++t) {
            const int i0 = F(t, 0), i1 = F(t, 1), i2 = F(t, 2);
            const Eigen::RowVector2d P0 = nodes2d.row(i0);
            const Eigen::RowVector2d P1 = nodes2d.row(i1);
            const Eigen::RowVector2d P2 = nodes2d.row(i2);
            const double two_area = std::abs(signed_double_area_2d(P0, P1, P2));

        for (const auto& q : qpts) {
            const double u_bary = 1.0 - q.v - q.w;
            const Eigen::RowVector2d Pq =
                u_bary * P0 + q.v * P1 + q.w * P2;
            x_q = Pq.transpose();

            const double w_dA = q.weight * two_area;  // physical area weight

            const auto gh = lme::evaluate_basis_grad_and_hess(
                nodes2d, beta, x_q, r_cut,
                params_.newton_tol, params_.newton_max_iters);

            const Eigen::Index n_act =
                static_cast<Eigen::Index>(gh.indices.size());

            // ----- bending block ----------------------------------------
            if (D_bend > 0.0) {
                // Stack each node's Hessian in Voigt form
                //   h_a = (H_xx, H_yy, H_xy)
                // into a 3 × n_act matrix; the per-(a, b) integrand is
                // h_a^T C_bend h_b, computed as one batched product.
                Eigen::MatrixXd H_voigt(3, n_act);
                for (Eigen::Index k = 0; k < n_act; ++k) {
                    const Eigen::MatrixXd& H = gh.hessians[
                        static_cast<std::size_t>(k)];
                    H_voigt(0, k) = H(0, 0);
                    H_voigt(1, k) = H(1, 1);
                    H_voigt(2, k) = H(0, 1);
                }
                const Eigen::MatrixXd CH    = C_bend * H_voigt;   // 3 × n_act
                const Eigen::MatrixXd Kloc =
                    (w_dA * D_bend) * (H_voigt.transpose() * CH);

                for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                    const int a = gh.indices[static_cast<std::size_t>(ia)];
                    for (Eigen::Index ib = 0; ib < n_act; ++ib) {
                        const int b = gh.indices[static_cast<std::size_t>(ib)];
                        const double v = Kloc(ia, ib);
                        if (v != 0.0) {
                            bend_acc[pair_key(a, b)] += v;
                        }
                    }
                }
            }

            // ----- membrane block ---------------------------------------
            // Strain-displacement matrix for each node (3 × 2 in Voigt
            // form on ε = (ε_xx, ε_yy, γ_xy)):
            //   B_a = [[N_a,x,    0     ],
            //          [   0   , N_a,y  ],
            //          [N_a,y , N_a,x  ]]
            // We stack B for all active nodes into a 3 × (2·n_act) block
            // B_all, then the per-pair (a, b) 2×2 stiffness block is the
            // corresponding slice of B_all^T · D_mem · B_all · k_L · w_dA.
            if (k_mem > 0.0) {
                Eigen::MatrixXd B_all(3, 2 * n_act);
                for (Eigen::Index k = 0; k < n_act; ++k) {
                    const Eigen::VectorXd& g = gh.gradients[
                        static_cast<std::size_t>(k)];
                    const double Nx = g(0);  // ∂N_a/∂x
                    const double Ny = g(1);  // ∂N_a/∂y
                    B_all(0, 2 * k + 0) = Nx;  B_all(0, 2 * k + 1) = 0.0;
                    B_all(1, 2 * k + 0) = 0.0; B_all(1, 2 * k + 1) = Ny;
                    B_all(2, 2 * k + 0) = Ny;  B_all(2, 2 * k + 1) = Nx;
                }
                const Eigen::MatrixXd DB    = D_mem * B_all;            // 3 × 2n_act
                const Eigen::MatrixXd Kloc =
                    (w_dA * k_mem) * (B_all.transpose() * DB);          // 2n_act × 2n_act

                for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                    const int a = gh.indices[static_cast<std::size_t>(ia)];
                    for (Eigen::Index ib = 0; ib < n_act; ++ib) {
                        const int b = gh.indices[static_cast<std::size_t>(ib)];
                        Eigen::Matrix2d block;
                        block(0, 0) = Kloc(2 * ia + 0, 2 * ib + 0);
                        block(0, 1) = Kloc(2 * ia + 0, 2 * ib + 1);
                        block(1, 0) = Kloc(2 * ia + 1, 2 * ib + 0);
                        block(1, 1) = Kloc(2 * ia + 1, 2 * ib + 1);
                        const std::int64_t key = pair_key(a, b);
                        auto it = mem_acc.find(key);
                        if (it == mem_acc.end()) {
                            mem_acc.emplace(key, block);
                        } else {
                            it->second += block;
                        }
                    }
                }
            }
        }

        // Emit one triplet per (a, b) entry in this triangle's
        // accumulators. trips_bend / trips_mem still need the 3·n_V
        // layout fix-up below; we keep the (a, b, scalar) and
        // (2a + di, 2b + dj, scalar) intermediate forms here for
        // continuity with the unchanged emission loop further down.
        }  // for t

        // Emit folded across this thread's whole triangle range (not per
        // triangle); setFromTriplets sums any cross-thread (a, b) splits.
        if (D_bend > 0.0) {
            for (const auto& [key, v] : bend_acc) {
                if (v == 0.0) continue;
                const int a = static_cast<int>(key / n_v);
                const int b = static_cast<int>(key % n_v);
                trips_bend.emplace_back(a, b, v);
            }
        }
        if (k_mem > 0.0) {
            for (const auto& [key, block] : mem_acc) {
                const int a = static_cast<int>(key / n_v);
                const int b = static_cast<int>(key % n_v);
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        const double v = block(di, dj);
                        if (v != 0.0) {
                            trips_mem.emplace_back(
                                2 * a + di, 2 * b + dj, v);
                        }
                    }
                }
            }
        }
    };  // worker

    run_threaded(n_threads, worker);

    // ----- assemble 3·n_v × 3·n_v K ----------------------------------
    // Layout: row/col 3a + d for vertex a, displacement component
    // d ∈ {0=x, 1=y, 2=z}. Bending lands on the z-block (d=2 only);
    // membrane lands on the (x, y) block (d ∈ {0, 1}). No cross-coupling
    // between in-plane and out-of-plane on a flat plate.
    //
    // Concatenate per-thread trips_bend / trips_mem buffers into a
    // single triplet list in the final 3·n_V row/column layout.
    std::size_t bend_total = 0, mem_total = 0;
    for (const auto& tt : bend_per_thread) bend_total += tt.size();
    for (const auto& tt : mem_per_thread)  mem_total  += tt.size();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(bend_total + mem_total);
    for (const auto& tt : bend_per_thread) {
        for (const auto& t : tt) {
            // (a, b, v) → (3a + 2, 3b + 2, v)
            trips.emplace_back(3 * t.row() + 2, 3 * t.col() + 2, t.value());
        }
    }
    for (const auto& tt : mem_per_thread) {
        for (const auto& t : tt) {
            // t.row() = 2a + di, t.col() = 2b + dj  →  (3a + di, 3b + dj, v)
            const int row2 = t.row();
            const int col2 = t.col();
            const int a    = row2 / 2;
            const int di   = row2 % 2;
            const int b    = col2 / 2;
            const int dj   = col2 % 2;
            trips.emplace_back(3 * a + di, 3 * b + dj, t.value());
        }
    }

    Eigen::SparseMatrix<double> K(3 * n_v, 3 * n_v);
    K.setFromTriplets(trips.begin(), trips.end());
    K.makeCompressed();
    // Finiteness backstop matching the curved path: a non-finite input
    // coordinate can reach K through the unguarded LDLT basis solve.
    if (!K.coeffs().allFinite()) {
        throw std::runtime_error(
            "LMEAssembler::assemble_K (flat): assembled stiffness matrix "
            "contains a non-finite entry — check the input geometry.");
    }
    return K;
}

Eigen::SparseMatrix<double> LMEAssembler::assemble_M(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double                 surface_density) const
{
    // ----- preconditions ---------------------------------------------
    if (V.rows() == 0 || F.rows() == 0) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_M: V and F must be non-empty");
    }
    if (!(surface_density > 0.0)) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_M: surface_density must be > 0 (got "
            + std::to_string(surface_density) + ")");
    }
    if (params_.use_second_order_sme && !params_.use_curved_shell) {
        throw std::invalid_argument(
            "LMEAssembler::assemble_M: use_second_order_sme requires "
            "use_curved_shell=true (the SME basis evaluates on the "
            "per-patch wPCA chart, which only exists on the curved path)");
    }

    // Curved-shell path: dispatch to the wPCA-patch-based assembler
    // (Millán 2011) when the caller has opted in. The curved path
    // reduces to the flat-plate consistent mass on planar input
    // (every chart's first fundamental form is identity and the
    // Shepard-PoU integration over input F triangles in R³ recovers
    // the same ρh·area = total mass).
    if (params_.use_curved_shell) {
        return assemble_M_curved(V, F, surface_density, params_);
    }

    require_flat_plate(V);

    // ----- geometry / locality ---------------------------------------
    const Eigen::Index n_v = V.rows();
    // 2D node positions for the LME basis (drop z; require_flat_plate
    // has confirmed it carries no information).
    Eigen::MatrixXd nodes2d = V.leftCols(2);

    const Eigen::VectorXd h_a = vertex_one_ring_h(V, F);
    Eigen::VectorXd beta(n_v);
    for (Eigen::Index a = 0; a < n_v; ++a) {
        beta(a) = params_.gamma / (h_a(a) * h_a(a));
    }

    // r_cut is uniform across the mesh — pick a value large enough that
    // every node has nontrivial basis support at any in-mesh query.
    // exp(-β r_cut²) is the tail magnitude; r_cut_mult ≥ 4 yields
    // exp(-1.6 · 16) ≈ 8e-12 at γ = 1.6, so partition-of-unity holds to
    // ~1e-11 at every quadrature point.
    const double r_cut = params_.r_cut_mult * h_a.maxCoeff();

    // ----- quadrature rule -------------------------------------------
    // Currently we always sample with 7-point Dunavant (consistent with
    // the LoopAssembler default); the @c n_quadrature_per_tri parameter
    // is recorded for future expansion but ignored here.
    const auto qpts =
        chladni::shell::quadrature_points(QuadratureRule::SevenPointDunavant);

    // ----- mass triplets ---------------------------------------------
    // M is laid out as 3·n_V × 3·n_V with three identical scalar blocks
    // (one per displacement component) — see header §dof. We build the
    // scalar block first; the 3-component tiling is folded into the
    // triplet list at the end.
    //
    // Per-cell accumulator: like the K path, all Gauss points within a
    // triangle contribute to the same (a, b) pairs, so we sum locally
    // and emit one triplet per (a, b) per triangle rather than per
    // (Gauss point × (a, b)).
    // Parallel triangle loop.
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_threads_raw = static_cast<int>(hw == 0 ? 1u : hw);
    const Eigen::Index n_F = F.rows();
    const int n_threads = static_cast<int>(
        std::min<Eigen::Index>(n_threads_raw, std::max<Eigen::Index>(1, n_F)));
    std::vector<std::vector<Eigen::Triplet<double>>> scalar_per_thread(
        static_cast<std::size_t>(n_threads));

    auto pair_key = [n_v_l = static_cast<std::int64_t>(n_v)](
                        int a, int b) -> std::int64_t {
        return static_cast<std::int64_t>(a) * n_v_l
             + static_cast<std::int64_t>(b);
    };

    auto worker = [&](int tid) {
        // Per-THREAD (a, b) accumulator — folds every triangle before emitting
        // (O(nnz), not O(nnz × support-triangles)). See assemble_K_curved_bending.
        std::unordered_map<std::int64_t, double> thread_acc;
        thread_acc.reserve(4096);
        auto& trips_scalar = scalar_per_thread[static_cast<std::size_t>(tid)];
        trips_scalar.reserve(static_cast<std::size_t>(n_F) * 144
                             / static_cast<std::size_t>(n_threads));

        const Eigen::Index chunk = (n_F + n_threads - 1) / n_threads;
        const Eigen::Index t_start = static_cast<Eigen::Index>(tid) * chunk;
        const Eigen::Index t_end   = std::min(t_start + chunk, n_F);

        Eigen::VectorXd x_q(2);
        for (Eigen::Index t = t_start; t < t_end; ++t) {
            const int i0 = F(t, 0), i1 = F(t, 1), i2 = F(t, 2);
            const Eigen::RowVector2d P0 = nodes2d.row(i0);
            const Eigen::RowVector2d P1 = nodes2d.row(i1);
            const Eigen::RowVector2d P2 = nodes2d.row(i2);
            // Twice the absolute area; multiplies the reference-triangle
            // weights (which themselves sum to 1/2).
            const double two_area = std::abs(signed_double_area_2d(P0, P1, P2));

        for (const auto& q : qpts) {
            // Barycentric → 2D position.
            const double u = 1.0 - q.v - q.w;
            const Eigen::RowVector2d Pq =
                u * P0 + q.v * P1 + q.w * P2;
            x_q = Pq.transpose();

            // ∫_T f dA = Σ_q w_q_ref · |2A_T| · f(x_q): reference rule
            // weights sum to 1/2 (the area of the unit-area triangle),
            // so multiplying by |2A_T| picks up exactly the physical
            // area scaling. (See `quadrature_points` in assembler.cpp.)
            const double w_pq =
                q.weight * two_area * surface_density;

            const auto bv = lme::evaluate_basis(
                nodes2d, beta, x_q, r_cut,
                params_.newton_tol, params_.newton_max_iters);

            const Eigen::Index n_act =
                static_cast<Eigen::Index>(bv.indices.size());
            for (Eigen::Index ia = 0; ia < n_act; ++ia) {
                const int    a   = bv.indices[static_cast<std::size_t>(ia)];
                const double p_a = bv.values[static_cast<std::size_t>(ia)];
                if (p_a == 0.0) continue;
                for (Eigen::Index ib = 0; ib < n_act; ++ib) {
                    const int    b   = bv.indices[
                        static_cast<std::size_t>(ib)];
                    const double p_b = bv.values[
                        static_cast<std::size_t>(ib)];
                    if (p_b == 0.0) continue;
                    thread_acc[pair_key(a, b)] += w_pq * p_a * p_b;
                }
            }
        }

        }  // for t

        // Emit one triplet per unique (a, b) folded across this thread's
        // entire range (not per triangle); setFromTriplets sums cross-thread
        // splits, and the ×3 tiling happens below.
        for (const auto& [key, v] : thread_acc) {
            if (v == 0.0) continue;
            const int a = static_cast<int>(key / n_v);
            const int b = static_cast<int>(key % n_v);
            trips_scalar.emplace_back(a, b, v);
        }
    };  // worker

    run_threaded(n_threads, worker);

    // Tile the scalar block onto all three displacement components.
    // The 3·n_V layout is `[u_0,x, u_0,y, u_0,z, u_1,x, ...]^T`, so a
    // scalar entry M_{a,b} contributes to rows/cols 3a+d, 3b+d for
    // d ∈ {0,1,2}. No cross-component coupling (Kirchhoff plate
    // decouples in-plane and out-of-plane motion at the mass level).
    std::size_t total_scalar = 0;
    for (const auto& tt : scalar_per_thread) total_scalar += tt.size();
    std::vector<Eigen::Triplet<double>> trips3;
    trips3.reserve(total_scalar * 3);
    for (const auto& tt : scalar_per_thread) {
        for (const auto& t3 : tt) {
            for (int d = 0; d < 3; ++d) {
                trips3.emplace_back(
                    3 * t3.row() + d, 3 * t3.col() + d, t3.value());
            }
        }
    }

    Eigen::SparseMatrix<double> M(3 * n_v, 3 * n_v);
    M.setFromTriplets(trips3.begin(), trips3.end());
    M.makeCompressed();
    // Finiteness backstop matching the curved path.
    if (!M.coeffs().allFinite()) {
        throw std::runtime_error(
            "LMEAssembler::assemble_M (flat): assembled mass matrix "
            "contains a non-finite entry — check the input geometry.");
    }
    return M;
}

Eigen::MatrixXd LMEAssembler::evaluate_modes_at_vertices(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& mode_coefficients) const
{
    if (!params_.use_ghost_nodes) {
        return mode_coefficients;
    }
    // Compute G from F to know how many ghost rows to drop. Cheap —
    // boundary-edge enumeration is O(F + V).
    const auto bdry = lme::collect_boundary_edges(F);
    const Eigen::Index G = static_cast<Eigen::Index>(bdry.size());
    if (G == 0) {
        return mode_coefficients;
    }
    const Eigen::Index n_v = V.rows();
    const Eigen::Index target_rows = 3 * n_v;
    if (mode_coefficients.rows() < target_rows) {
        throw std::invalid_argument(
            "LMEAssembler::evaluate_modes_at_vertices: mode_coefficients "
            "has fewer rows than 3*V.rows(); cannot slice");
    }

    // R2 (review 2026-06-09): the review asked for an EXACT basis projection
    // u(x_j) = Σ_a N_a(x_j) c_a here instead of slicing the real-vertex
    // coefficients, on the assumption that the max-ent basis is not
    // interpolating. Measured (test_lme_evaluate_modes_projection.cpp): the
    // COMPOSITE curved LME basis IS interpolating at the mesh vertices, so
    // the slice already IS the exact projection. Why: at a vertex x_j the
    // Shepard PoU activates the anchor patch j and its neighbours, but x_j
    // projects onto node j's OWN chart position in EVERY chart that contains
    // j (the projection is the same map that placed node j there), so each
    // active patch's in-chart solve hits its exact-node match and returns
    // {j: 1}. Summed by the partition of unity, N_a(x_j) = δ_aj exactly.
    // Building the composite evaluator here (it ran with zero throws) and
    // applying it reproduced the sliced coefficients bit-for-bit, so the
    // projection loop was pure cost — reverted. The slice is correct.
    return mode_coefficients.topRows(target_rows);
}

std::string LMEAssembler::label() const
{
    return "LME (Arroyo-Ortiz)";
}

}  // namespace chladni::shell
