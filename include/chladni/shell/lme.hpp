#pragma once

/**
 * @file lme.hpp
 * @brief Local Maximum-Entropy (LME) meshfree basis and shell assembler.
 *
 * Declarations for the **1st-order Local Max-Ent approximant** of
 * @cite arroyo_ortiz_2006_local_maximum_entropy and the matching
 * @ref chladni::shell::LMEAssembler that plugs into the polymorphic
 * @ref chladni::shell::ShellAssembler interface.
 *
 * @section lme_algorithm Algorithm at a glance
 *
 * Given a node set @f$ X = \{x_a\}_{a=1}^N \subset \mathbb{R}^d @f$ and
 * per-node locality parameters @f$ \beta_a > 0 @f$, the LME basis
 * values at a query point @f$ x \in \mathrm{conv}\,X @f$ solve
 *
 *   @f[
 *     \min_{p \in \mathbb{R}^N}
 *       \sum_a \beta_a\, p_a |x - x_a|^2  +  \sum_a p_a \ln p_a
 *     \quad\text{s.t.}\quad
 *     p_a \ge 0,\ \sum_a p_a = 1,\ \sum_a p_a x_a = x.
 *   @f]
 *
 * The convex dual is unconstrained in @f$ \lambda \in \mathbb{R}^d @f$:
 *
 *   @f[
 *     \lambda^\star(x) = \arg\min_{\lambda \in \mathbb{R}^d} \ln Z(x, \lambda),
 *     \quad
 *     Z(x, \lambda) = \sum_a \exp\!\bigl(
 *       -\beta_a |x - x_a|^2 + \lambda \cdot (x - x_a)
 *     \bigr).
 *   @f]
 *
 * Newton's method converges quadratically from @f$\lambda = 0@f$
 * (5--10 iterations typical). The basis is then
 *
 *   @f[
 *     p_a(x) =
 *       \frac{\exp\!\bigl(-\beta_a |x-x_a|^2 + \lambda^\star \cdot (x-x_a)\bigr)}
 *            {Z(x, \lambda^\star)} .
 *   @f]
 *
 * Properties (proved in Arroyo--Ortiz 2006):
 *
 *  - @f$ p_a(x) \ge 0 @f$ everywhere, @f$ \sum_a p_a(x) = 1 @f$
 *    (partition of unity).
 *  - Affine reproduction: @f$ \sum_a p_a(x)\, x_a = x @f$.
 *  - C@f$^\infty@f$ smooth in the interior of @f$ \mathrm{conv}\,X @f$.
 *  - **Weak Kronecker-delta at corners** of @f$ \mathrm{conv}\,X @f$:
 *    if @f$ x_b @f$ is a vertex of the convex hull then
 *    @f$ p_a(x_b) = \delta_{ab} @f$.
 *
 * @section lme_status Current status
 *
 * The 1st-order LME basis evaluator (Newton solver on
 * @f$ \lambda^\star @f$), its closed-form gradient and Hessian, and the
 * full @ref chladni::shell::LMEAssembler are all shipped. Galerkin mass
 * (@ref chladni::shell::LMEAssembler::assemble_M) integrates the consistent
 * @f$ M_{ab} = \int \rho h\, p_a p_b\,dA @f$. Stiffness
 * (@ref chladni::shell::LMEAssembler::assemble_K) assembles the
 * Millan 2011 curved-shell Kirchhoff-Love energy (wPCA charts +
 * Shepard partition of unity) by default, or — with curved support
 * off — flat-plate Kirchhoff bending on the z component and 2D
 * plane-stress membrane on the @f$(u_x, u_y)@f$ components. Either
 * way the kernel is the 6-dim rigid-body subspace, matching
 * @ref chladni::shell::LoopAssembler. The 2nd-order SME basis
 * (@cite rosolen_millan_arroyo_2013_second_order_maxent) is shipped
 * as an alternative basis on the same assembly path.
 *
 * @section lme_refs References
 *
 *  - @cite arroyo_ortiz_2006_local_maximum_entropy — foundational
 *    construction.
 *  - @cite rosolen_millan_arroyo_2013_second_order_maxent — 2nd-order
 *    SME fallback and biharmonic shell-vibration validation.
 *  - @cite millan_rosolen_arroyo_2011_thin_shell_maxent — curved-shell
 *    wPCA + Shepard partition-of-unity machinery (later milestone).
 *
 * @see chladni::shell::ShellAssembler — the polymorphic interface.
 * @see chladni::shell::LoopAssembler  — the existing Loop FEM baseline.
 */

#include <chladni/shell/assembler.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <string>
#include <vector>

namespace chladni::shell {

/**
 * @brief LME basis values at a single query point.
 *
 * Sparse representation. The @ref indices vector lists the node ids
 * whose basis values are above the truncation threshold (or all nodes
 * if @c r_cut covers the entire node set, e.g. in unit tests). The
 * @ref values vector holds the corresponding @f$ p_a(x) \in [0, 1] @f$.
 * Both vectors have identical length; entry @c k refers to node
 * @c indices[k] with value @c values[k].
 *
 * Properties guaranteed by @ref lme::evaluate_basis on successful
 * Newton convergence:
 *  - Partition of unity: @f$ \sum_k \mathrm{values}[k] = 1 @f$ to
 *    machine precision.
 *  - Affine reproduction: @f$ \sum_k \mathrm{values}[k]\, x_{\mathrm{indices}[k]} = x @f$
 *    to the configured Newton tolerance.
 *  - Nonnegativity: @f$ \mathrm{values}[k] \ge 0 @f$ for all @c k.
 */
struct LMEBasisValues {
    /// Node ids referenced by the active set (entries into the
    /// caller's @c nodes matrix).
    std::vector<int>    indices;

    /// Basis values @f$ p_a(x) @f$ for @c a in @ref indices; same
    /// length as @ref indices.
    std::vector<double> values;
};

/**
 * @brief LME basis values **and gradients** at a single query point.
 *
 * Adds per-node spatial gradients to @ref LMEBasisValues. The closed
 * form (Arroyo--Ortiz 2006 eq. 44 with uniform @f$ \beta @f$):
 *
 *   @f[
 *     \nabla p_a(x) = -p_a(x)\, J^{-1}(x)\, (x - x_a),
 *     \qquad J = \sum_b p_b(x)\, (x - x_b)(x - x_b)^\top .
 *   @f]
 *
 * Consistent reproduction identities (used by the unit tests):
 *  - @f$ \sum_a \nabla p_a(x) = 0 @f$ (differentiating the PoU
 *    identity @f$ \sum_a p_a = 1 @f$).
 *  - @f$ \sum_a (x_a)\otimes (\nabla p_a)^\top = I @f$
 *    (differentiating the linear-reproduction identity
 *    @f$ \sum_a p_a\,x_a = x @f$).
 */
struct LMEBasisAndGrad {
    /// Node ids referenced by the active set.
    std::vector<int>             indices;

    /// Basis values @f$ p_a(x) @f$, length @c indices.size().
    std::vector<double>          values;

    /// Basis gradients @f$ \nabla p_a(x) \in \mathbb R^d @f$, one
    /// entry per active node. Same length as @ref indices.
    std::vector<Eigen::VectorXd> gradients;
};

/**
 * @brief LME basis values, gradients, and **Hessians** at a single
 *        query point.
 *
 * Adds per-node spatial Hessians to @ref LMEBasisAndGrad. With
 * @f$ v_a = J^{-1} (x - x_a) @f$ and
 * @f$ \alpha_{ba} = (x - x_b)^\top v_a @f$, differentiating the
 * gradient closed form @f$ \nabla p_a = -p_a\, v_a @f$ once more
 * yields (uniform @f$\beta@f$)
 *
 *   @f[
 *     \nabla^2 p_a(x)
 *     = p_a\!\left(
 *         v_a v_a^\top
 *         \;-\;\sum_b p_b\,\alpha_{ba}\,v_b v_b^\top
 *         \;-\;J^{-1}
 *       \right).
 *   @f]
 *
 * This matches the Millan 2011 eq. A7 expression (with the
 * @f$ \nabla \beta @f$ term dropped for spatially-constant
 * @f$ \beta @f$).
 *
 * Consistency identities (used by the unit tests):
 *  - @f$ \sum_a \nabla^2 p_a(x) = 0 @f$ (differentiating
 *    @f$ \sum_a \nabla p_a = 0 @f$).
 *  - @f$ \sum_a x_{a,k}\,\nabla^2 p_a(x) = 0 @f$ for each
 *    component @f$ k @f$ (differentiating
 *    @f$ \sum_a x_a \otimes (\nabla p_a)^\top = I @f$, which is
 *    constant in @f$ x @f$).
 */
struct LMEBasisGradHess {
    /// Node ids referenced by the active set.
    std::vector<int>             indices;

    /// Basis values @f$ p_a(x) @f$.
    std::vector<double>          values;

    /// Basis gradients @f$ \nabla p_a(x) \in \mathbb R^d @f$.
    std::vector<Eigen::VectorXd> gradients;

    /// Basis Hessians @f$ \nabla^2 p_a(x) \in \mathbb R^{d \times d} @f$,
    /// each symmetric.
    std::vector<Eigen::MatrixXd> hessians;
};

namespace lme {

/**
 * @brief Local geometric chart at a patch anchor point @f$ Q_A @f$.
 *
 * Encapsulates the weighted-PCA construction of
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent §2.1 — the
 * first stage of curved-shell LME assembly. For each anchor on the
 * input point cloud we build:
 *
 *  - The Gaussian-weighted centroid @f$ \bar Q_A = \sum_a w_a^P(Q_A)\, P_a @f$
 *    of the participating neighbourhood.
 *  - The 3×3 weighted covariance
 *    @f$ C_A = X_A\,\mathrm{diag}(w_a^P(Q_A))\,X_A^\top @f$
 *    where the columns of @f$ X_A @f$ are
 *    @f$ P_a - \bar Q_A @f$.
 *  - The two-dimensional tangent frame @f$ V_A \in \mathbb R^{3\times 2} @f$
 *    whose columns are the eigenvectors of @f$ C_A @f$ associated with
 *    its two largest eigenvalues.
 *  - The orthogonal projection of each neighbour into the tangent plane,
 *    @f$ \xi_a = V_A^\top (P_a - \bar Q_A) \in \mathbb R^2 @f$ — the
 *    chart coordinates that feed the in-chart 2D LME basis solve.
 *
 * The smallest eigenvalue @ref out_of_plane_eig is recorded as a
 * diagnostic for callers reasoning about local curvature error (a small
 * value confirms the neighbourhood is nearly planar, so the tangent chart
 * is a faithful flattening). It is currently informational only — the
 * assembler does not gate on it.
 *
 * Sign convention for @f$ V_A @f$: the columns of @f$ V_A @f$ are
 * orthonormal but their individual orientation is not pinned —
 * Eigen's @c SelfAdjointEigenSolver returns eigenvectors with
 * arbitrary sign. Callers using @c xi or the tangent frame @ref V
 * see a consistent set of coordinates per Patch, but two independent
 * @ref build_patch calls on slightly perturbed inputs may flip signs.
 * All downstream Galerkin assembly is invariant to this sign choice
 * (it only sees @f$ \xi_a^\top \cdot \xi_b @f$ and
 * @f$ V_A V_A^\top @f$, both sign-stable).
 */
struct Patch {
    /// The patch label @f$ A @f$ — index of the anchor in the parent
    /// node set @f$ P @f$.
    int                                       anchor_id = -1;

    /// Anchor position @f$ Q_A = P_{\text{anchor\_id}} \in \mathbb R^3 @f$.
    Eigen::Vector3d                           Q;

    /// Gaussian-weighted centroid
    /// @f$ \bar Q_A = \sum_a w_a^P(Q_A) P_a @f$. Coincides with @ref Q
    /// only when the neighbourhood is symmetric about @ref Q.
    Eigen::Vector3d                           Qbar;

    /// Tangent frame @f$ V_A \in \mathbb R^{3 \times 2} @f$. Columns
    /// are the two unit eigenvectors of the weighted covariance with
    /// the largest eigenvalues — orthonormal but with arbitrary sign.
    Eigen::Matrix<double, 3, 2>               V;

    /// Smallest eigenvalue of the weighted covariance — the
    /// out-of-plane spread of the neighbourhood relative to the
    /// tangent plane @ref V. Zero iff the neighbourhood is exactly
    /// coplanar with @ref Qbar.
    double                                    out_of_plane_eig = 0.0;

    /// Indices into the parent node set of the neighbours that
    /// participate in this chart. Order matches @ref xi rows.
    std::vector<int>                          neighbor_ids;

    /// Per-neighbour 2D projections
    /// @f$ \xi_a = V_A^\top (P_a - \bar Q_A) @f$; one row per entry in
    /// @ref neighbor_ids. Stored as @c Eigen::MatrixXd (Dynamic ×
    /// Dynamic) rather than @c Matrix<double, Dynamic, 2> so it binds
    /// directly to the @c const MatrixXd& parameter of
    /// @ref evaluate_basis without an Eigen converting-copy at every
    /// in-chart basis call (the assembly inner loop hits this fix
    /// 10^5+ times per K assembly on a moderate mesh).
    Eigen::MatrixXd                           xi;
};

/**
 * @brief Build a tangent chart at @p anchor_id via weighted PCA.
 *
 * Runs the chart construction of @ref Patch on the neighbourhood
 * @p neighbor_ids. Gaussian weights are
 *
 *   @f[
 *     w_a = \exp\!\bigl(-\beta_a |P_a - Q_A|^2\bigr),
 *   @f]
 *
 * normalised by @f$ \sum_a w_a @f$ to give partition-of-unity weights
 * (the Shepard PU in
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent eq. 1 specialised
 * to @f$ x = Q_A @f$ and the neighbourhood @p neighbor_ids). The
 * normalisation has no effect on the principal axes (the covariance is
 * invariant under a uniform rescaling of weights) — it only sets the
 * scale of the eigenvalues @ref Patch::out_of_plane_eig.
 *
 * @param anchor_id   Index of the patch anchor in @p nodes; must
 *                    satisfy @c 0 <= anchor_id < nodes.rows().
 * @param nodes       @f$ N \times 3 @f$ row matrix of node positions
 *                    in @f$ \mathbb R^3 @f$.
 * @param neighbor_ids Indices of the participating neighbours. May or
 *                    may not contain @p anchor_id; including it is the
 *                    typical case (the anchor is one of its own
 *                    neighbours in @c Q=P operation).
 * @param beta_wpca   Per-node Gaussian decay rates @f$ \beta_a > 0 @f$
 *                    of length @c nodes.rows(). For the standard
 *                    aspect-ratio convention this is
 *                    @f$ \beta_a = \gamma_{\text{wPCA}} / h_a^2 @f$.
 *
 * @throws std::invalid_argument if @p anchor_id is out of range, the
 *         neighbour list is empty, any @f$ \beta_a \le 0 @f$, or the
 *         neighbour set degenerates (every weight is zero in floating
 *         point — typical when @c beta_wpca is wildly too large).
 */
[[nodiscard]] Patch build_patch(
    int                       anchor_id,
    const Eigen::MatrixXd&    nodes,
    const std::vector<int>&   neighbor_ids,
    const Eigen::VectorXd&    beta_wpca);

/**
 * @brief Restrict a chart node selection to the anchor's geodesically
 *        connected component.
 *
 * The geometric (value-based) chart selection on the curved LME path
 * realises the paper's metric-ball neighbourhoods
 * (@cite millan_rosolen_arroyo_2011_thin_shell_maxent Eq. 2). A pure
 * metric ball on a sampled surface is only a topological DISK under
 * the paper's sampling assumption — applicability "depends crucially
 * … on the density of the sampling relative to the feature size"
 * (their §5). On arbitrary user meshes that assumption can fail: two
 * nearby sheets of a fold may both enter the ball while the path
 * connecting them exits it, leaving the selection with several
 * connected components. wPCA-projecting such an island onto the
 * anchor's tangent plane folds it into spurious chart positions and
 * couples geodesically distant surface regions through the basis.
 *
 * This helper enforces the disk property BY CONSTRUCTION: a
 * breadth-first search from @p anchor through @p adjacency restricted
 * to the selected @p nodes, keeping only the nodes reachable within
 * the selection (relative input order preserved). Selections that are
 * already connected — every chart on a sampling-adequate mesh — pass
 * through unchanged.
 *
 * @param anchor    Index of the chart anchor; always kept. Must be
 *                  contained in @p nodes for a non-trivial result.
 * @param nodes     The selected chart node indices (anchor included).
 * @param adjacency Vertex-vertex adjacency of the (ghost-augmented)
 *                  node set; entry @c v lists the neighbours of @c v.
 *
 * @return The subsequence of @p nodes reachable from @p anchor inside
 *         the selection. If @p anchor is not in @p nodes, returns
 *         just @c {anchor}.
 */
[[nodiscard]] std::vector<int> prune_to_anchor_component(
    int                                  anchor,
    std::vector<int>                     nodes,
    const std::vector<std::vector<int>>& adjacency);

/**
 * @brief Chart neighbourhood of @p anchor: the connected component of
 *        the metric ball containing the anchor, via in-ball BFS.
 *
 * Realises the paper's value-based neighbourhood
 * (@cite millan_rosolen_arroyo_2011_thin_shell_maxent Eq. 2 — all
 * nodes within a Euclidean radius) WITH the topological-disk property
 * the paper otherwise assumes from sampling density (their §5),
 * obtained by construction: BFS from the anchor that only expands
 * through in-ball nodes. A disconnected island (Euclidean-near nodes
 * whose connecting path exits the ball, e.g. the second sheet of a
 * fold) is unrepresentable — the search cannot reach it — and a
 * geodesically-far-but-Euclidean-near node (e.g. a sphere antipode)
 * is excluded for the same reason, which retires the earlier
 * fixed-depth BFS guard. Unlike that guard, the ball is not hop-
 * bounded: on strongly graded meshes in-ball nodes more than 8 hops
 * out are (correctly, per Eq. 2) included.
 *
 * Post-processing, in order:
 *  - if fewer than @p min_nodes survive, fall back to the plain
 *    2-ring (connected by construction) so coarse/boundary anchors
 *    are never starved below 2nd-order wPCA/basis support;
 *  - if more than @p max_nodes survive, keep the @p max_nodes
 *    nearest (bounds the O(n²) in-chart Newton; prevents the
 *    high-valence freeze) and re-run
 *    @ref prune_to_anchor_component — the nearest-cap can in
 *    principle disconnect the kept subset.
 *
 * @param anchor    Chart anchor index; always first in the result.
 * @param radius    Metric-ball radius (absolute, same units as
 *                  @p nodes). Callers derive it from the per-node
 *                  spacing, e.g. @f$ h_a\sqrt{\ln(1/tol)/\gamma} @f$.
 * @param nodes     @f$ N \times 3 @f$ node positions (ghost-extended).
 * @param adjacency Vertex-vertex adjacency of the same node set.
 * @param min_nodes Minimum chart size before the 2-ring fallback.
 * @param max_nodes Hard cap on the chart size.
 *
 * @return Chart node indices in BFS order (anchor first).
 */
[[nodiscard]] std::vector<int> extract_chart_neighbourhood(
    int                                  anchor,
    double                               radius,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    int                                  min_nodes,
    int                                  max_nodes);

/**
 * @brief Anisotropic (directional-spacing) variant of
 *        @ref extract_chart_neighbourhood for the 2nd-order SME path.
 *
 * @note RETAINED-REFERENCE EXPERIMENT — not wired to any assembly path
 *       or @c Params knob, reachable only from its unit test. This is
 *       the round-2 chart extractor of the SME chart A/B (scaffolding
 *       inventory C4): it lost the cylinder aspect ladder at every
 *       level and was superseded by the round-3 intrinsic extractor
 *       (@ref extract_chart_neighbourhood_intrinsic, behind
 *       @c Params::sme_chart_radius_mult). Kept, with its spec test, as
 *       the documented record of that A/B; delete if the A/B record is
 *       no longer wanted.
 *
 * Same connected in-ball BFS construction, but the acceptance region
 * is shaped by the anchor's DIRECTIONAL nodal spacing rather than an
 * isotropic radius: candidate @c v at offset @f$ y = x_v - x_a @f$ is
 * in-range iff
 * @f[
 *   \|y\| \;\le\; \mathrm{mult}\cdot h_a(\hat y),
 *   \qquad h_a(u) = \max_b |(x_b - x_a)\cdot u|
 * @f]
 * — i.e. the distance toward @c v, measured in units of the one-ring
 * spacing IN THAT DIRECTION, is at most @p mult (implemented as
 * @f$ \|y\|^2 \le \mathrm{mult}\cdot\max_b|e_b\cdot y| @f$, no
 * normalisation needed). @f$ h_a(u) @f$ is the same faithful
 * max-projection spacing the SME gap matrices use
 * (@cite rosolen_millan_arroyo_2013_second_order_maxent §3.2) — chart
 * support and slack thus share one geometric object, and the chart
 * reaches ~@p mult spacings in every direction regardless of mesh
 * anisotropy (an ISOTROPIC ball on an anisotropic grid spans a wide
 * arc in the fine direction — measured +349 % on the aspect-2.39
 * cylinder — which is exactly what this variant avoids; @p mult ≈ 3
 * mimics the legacy k-ring-3 extent). Two free properties:
 *  - on a flat sheet the normal-direction spacing is exactly zero, so
 *    a fold's second sheet is metrically out of range with no
 *    epsilon floor;
 *  - on a curved surface the curvature sag provides the natural
 *    normal-direction scale.
 * The nearest-cap orders by the same directional measure
 * @f$ \|y\|^2 / \max_b|e_b\cdot y| @f$.
 *
 * @param anchor       Chart anchor index (a REAL vertex; ghosts do
 *                     not anchor patches).
 * @param mult         Reach in directional-spacing units (~3).
 * @param anchor_edges One-ring edge vectors @f$ x_b - x_a @f$ of the
 *                     anchor (world coordinates).
 * @param nodes        @f$ N \times 3 @f$ node positions
 *                     (ghost-extended).
 * @param adjacency    Vertex-vertex adjacency of the same node set.
 * @param min_nodes    Minimum chart size before the 2-ring fallback.
 * @param max_nodes    Hard cap on the chart size.
 *
 * @return Chart node indices in BFS order (anchor first).
 */
[[nodiscard]] std::vector<int> extract_chart_neighbourhood_directional(
    int                                  anchor,
    double                               mult,
    const std::vector<Eigen::Vector3d>&  anchor_edges,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    int                                  min_nodes,
    int                                  max_nodes);

/**
 * @brief INTRINSIC (geodesic) variant of
 *        @ref extract_chart_neighbourhood — Dijkstra over mesh edge
 *        lengths, keeping every node whose along-surface distance to
 *        the anchor is at most @p radius.
 *
 * The k-ring measures hops, the extrinsic variants measure CHORDS
 * through ambient space; neither is the quantity that controls chart
 * quality. What wPCA projection actually requires is that the chart
 * span a small GEODESIC neighbourhood (Millán 2011 §2.3's "small
 * neighborhood", §5's projections that "should not distort too much
 * the node geometry") — and the graph shortest-path length over edge
 * lengths is its standard discrete approximation. Two properties come
 * for free:
 *  - a fold's second sheet, or a sphere antipode, is intrinsically
 *    FAR no matter how Euclidean-near, so no connectivity machinery
 *    is needed: a Dijkstra ball is connected by construction (every
 *    prefix of a shortest path is in the ball);
 *  - "around the curve" costs arc length rather than chord length,
 *    removing the extrinsic ball's arc-folding failure mode on
 *    coarse curved meshes.
 *
 * Post-processing matches the other extractors: 2-ring fallback below
 * @p min_nodes; above @p max_nodes keep the intrinsically nearest
 * (still a Dijkstra ball, hence still connected; the re-prune is kept
 * as a cheap unconditional guarantee).
 *
 * SCALING NOTE: callers currently pass @c radius = mult·h_a(anchor),
 * which (like the k-ring and the extrinsic variants) shrinks with
 * subdivision. The intrinsic form is the one for which an ABSOLUTE or
 * curvature-tied radius (charts covering a fixed geodesic patch,
 * gaining nodes under refinement) is also meaningful — that scaling
 * choice is deliberately left to the caller.
 *
 * @param anchor    Chart anchor index; always first in the result.
 * @param radius    Geodesic radius (absolute length units).
 * @param nodes     @f$ N \times 3 @f$ node positions (ghost-extended).
 * @param adjacency Vertex-vertex adjacency of the same node set.
 * @param min_nodes Minimum chart size before the 2-ring fallback.
 * @param max_nodes Hard cap on the chart size.
 *
 * @return Chart node indices ordered by intrinsic distance
 *         (anchor first).
 */
[[nodiscard]] std::vector<int> extract_chart_neighbourhood_intrinsic(
    int                                  anchor,
    double                               radius,
    const Eigen::MatrixXd&               nodes,
    const std::vector<std::vector<int>>& adjacency,
    int                                  min_nodes,
    int                                  max_nodes);

/**
 * @brief Shepard partition-of-unity weights at a single query point.
 *
 * Sparse representation of the @f$ w_A^Q(x) @f$ values from
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent eqs. (1)–(3).
 * Entries are post-truncation and post-renormalisation: any patch
 * whose pre-truncation weight falls below the caller-supplied
 * @c tol is dropped, then the surviving weights are rescaled so
 * @f$ \sum_A w_A^Q(x) = 1 @f$ exactly.
 */
struct ShepardWeights {
    /// Indices into the patch-point set @f$ Q = \{Q_A\} @f$ of the
    /// patches with above-tolerance support at the query point.
    std::vector<int>    indices;

    /// Corresponding weights @f$ w_A^Q(x) @f$; length matches
    /// @ref indices.
    std::vector<double> values;
};

/**
 * @brief Evaluate Shepard partition-of-unity weights at @p x.
 *
 * @f[
 *   w_A^Q(x)
 *   = \frac{\exp\!\bigl(-\beta_A |x - Q_A|^2\bigr)}
 *          {\sum_B \exp\!\bigl(-\beta_B |x - Q_B|^2\bigr)} .
 * @f]
 *
 * Implementation uses the standard log-sum-exp shift to avoid
 * underflow at large @f$ \beta @f$ or far queries. Patches whose
 * normalised weight is @f$ \le @f$ @p tol are dropped, and the
 * surviving weights are renormalised to sum to 1.
 *
 * @param patch_points  @f$ M \times 3 @f$ row matrix of patch anchor
 *                      positions @f$ Q_A \in \mathbb R^3 @f$.
 * @param beta_patches  Length-@f$ M @f$ vector of per-patch Gaussian
 *                      decay rates @f$ \beta_A > 0 @f$. Typical
 *                      @f$ \beta_A = \gamma_{\mathrm{PU}} / h_A^2 @f$
 *                      with paper default
 *                      @f$ \gamma_{\mathrm{PU}} \in [3, 6] @f$.
 * @param x             Query point @f$ x \in \mathbb R^3 @f$.
 * @param tol           Drop patches whose normalised weight falls at
 *                      or below this value. Strictly positive.
 *
 * @throws std::invalid_argument if @p patch_points is empty,
 *         @c beta_patches.size() != patch_points.rows(), any
 *         @f$ \beta_A \le 0 @f$, or @p tol is non-positive.
 */
[[nodiscard]] ShepardWeights shepard_partition(
    const Eigen::MatrixXd&                      patch_points,
    const Eigen::VectorXd&                      beta_patches,
    const Eigen::Ref<const Eigen::Vector3d>&    x,
    double                                      tol = 1.0e-10);

/**
 * @brief Curved-shell LME basis values at a query point @f$ y @f$.
 *
 * Sparse representation of the per-global-node weights
 *
 *   @f[
 *     T_a(y)
 *     = \sum_{A \in \mathcal N_y^Q}
 *         w_A^Q(y)\;
 *         p_a\!\bigl(\Pi_A(y)\bigr) ,
 *   @f]
 *
 * the closed form of
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent eq. (8) applied to
 * a single field component. A displacement field
 * @f$ \{u_a\}_{a=1}^N @f$ is reconstructed at @f$ y \in \mathcal M @f$
 * as @f$ u_h(y) = \sum_a T_a(y)\, u_a @f$. The weights inherit the
 * partition-of-unity property from both factors:
 *
 *   @f[
 *     \sum_a T_a(y) = \sum_A w_A^Q(y)\, \sum_a p_a(\Pi_A(y)) = 1 .
 *   @f]
 */
struct CurvedBasisWeights {
    /// Indices into the parent node set @f$ P @f$.
    std::vector<int>    indices;

    /// Composite weights @f$ T_a(y) @f$; same length as @ref indices.
    std::vector<double> values;
};

/**
 * @brief Evaluate the curved-shell LME basis at @p y.
 *
 * Walks the active Shepard patches at @p y (via @ref shepard_partition
 * on @p patch_points), projects @p y into each chart through @ref Patch::V
 * and the chart's centroid @c Qbar, evaluates the 2D in-chart LME basis
 * on the chart's projected node set @ref Patch::xi, and accumulates the
 * Shepard-weighted contributions into a sparse map over the parent
 * node ids.
 *
 * @param patches         Per-patch tangent charts built once via
 *                        @ref build_patch; one entry per row of
 *                        @p patch_points.
 * @param patch_points    @f$ M \times 3 @f$ row matrix of patch anchor
 *                        positions @f$ Q_A @f$.
 * @param beta_patches    Per-patch Shepard decay rates @f$ \beta_A @f$.
 * @param nodes           @f$ N \times 3 @f$ row matrix of parent node
 *                        positions; used only for the per-node LME
 *                        β bookkeeping via @p beta_lme indexing.
 * @param beta_lme        Per-global-node LME decay rates
 *                        @f$ \beta_a = \gamma_{\mathrm{LME}} / h_a^2 @f$.
 *                        Each patch evaluates the in-chart basis at the
 *                        slice of @p beta_lme matching its
 *                        @c neighbor_ids.
 * @param y               3D query point.
 * @param tol_shepard     Truncation tolerance for the Shepard PU; see
 *                        @ref shepard_partition.
 * @param r_cut           In-chart truncation radius for the 2D LME
 *                        basis; same semantics as
 *                        @ref evaluate_basis. Pass a generous value
 *                        (larger than the largest patch extent) to
 *                        keep every neighbour active.
 * @param newton_tol      Newton tolerance for the in-chart basis solve.
 * @param newton_max_iters Newton iteration cap for the in-chart solve.
 *
 * @throws std::invalid_argument if @p patches is empty,
 *         @c patch_points.rows() != patches.size(),
 *         @p nodes is empty, or any
 *         @c neighbor_ids entry of a participating patch falls outside
 *         the @p beta_lme range.
 * @throws std::runtime_error if the in-chart Newton fails to converge
 *         for an active patch.
 */
[[nodiscard]] CurvedBasisWeights evaluate_basis_curved(
    const std::vector<Patch>&                   patches,
    const Eigen::MatrixXd&                      patch_points,
    const Eigen::VectorXd&                      beta_patches,
    const Eigen::MatrixXd&                      nodes,
    const Eigen::VectorXd&                      beta_lme,
    const Eigen::Ref<const Eigen::Vector3d>&    y,
    double tol_shepard      = 1.0e-10,
    double r_cut            = 1.0e6,
    double newton_tol       = 1.0e-10,
    int    newton_max_iters = 30);

/**
 * @brief Evaluate the 1st-order Local Max-Ent basis at @p x.
 *
 * Solves the dual Newton minimisation for @f$ \lambda^\star(x) @f$
 * and returns the basis values @f$ p_a(x) @f$ for every node within
 * truncation radius @p r_cut of @p x.
 *
 * @param nodes
 *   @f$ N \times d @f$ row matrix of node positions
 *   @f$ x_a \in \mathbb{R}^d @f$. Currently used with @f$ d = 2 @f$
 *   (flat plates); curved-shell support comes via per-patch tangent
 *   projection in a later milestone.
 * @param beta
 *   Length-@f$ N @f$ vector of per-node locality parameters
 *   @f$ \beta_a > 0 @f$. Practical choice
 *   @f$ \beta_a = \gamma / h_a^2 @f$ with @f$ h_a @f$ the local node
 *   spacing and @f$ \gamma \approx 1.6 @f$ (Millan 2011 default).
 * @param x
 *   Query point @f$ x \in \mathbb{R}^d @f$. Must lie in
 *   @f$ \mathrm{conv}\,X @f$ for Newton to converge.
 * @param r_cut
 *   Truncation radius. Nodes with @f$ |x - x_a| > r_{\mathrm{cut}} @f$
 *   are excluded from the active set. Set arbitrarily large to disable
 *   truncation (useful for unit tests where strict partition of unity
 *   is required).
 * @param newton_tol
 *   Convergence tolerance on @f$ \|\nabla_\lambda \ln Z\|_\infty @f$.
 *   @c 1e-10 is the shipped default; tighter is rarely useful.
 * @param newton_max_iters
 *   Hard upper bound on Newton iterations; throws @c std::runtime_error
 *   if not reached. Typical convergence is 5--10 from @f$ \lambda = 0 @f$.
 *
 * @return Basis values at @p x; see @ref LMEBasisValues for guarantees.
 *
 * @throws std::invalid_argument if @p nodes is empty,
 *   @c beta.size() != nodes.rows(), any @f$ \beta_a \le 0 @f$, or
 *   the active set after @p r_cut truncation is empty.
 * @throws std::runtime_error if Newton does not converge within
 *   @p newton_max_iters iterations.
 *
 * @note If @p x coincides with a node @f$ x_b @f$ to within
 *   @f$ 10^{-12} @f$ (relative to the local @f$ h_a @f$), the routine
 *   returns @f$ p_b = 1 @f$ and all other entries zero, bypassing
 *   Newton. This handles the conv-hull-corner case (where the dual
 *   Hessian degenerates) cleanly; the weak Kronecker-delta property is
 *   reproduced exactly.
 */
[[nodiscard]] LMEBasisValues evaluate_basis(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 30);

/**
 * @brief Evaluate LME basis values **and** spatial gradients at @p x.
 *
 * Shares the dual Newton solve with @ref evaluate_basis; after Newton
 * converges, the gradients are filled in via the Arroyo--Ortiz 2006
 * eq. 44 closed form
 * @f$ \nabla p_a = -p_a J^{-1} (x - x_a) @f$ (uniform-@f$\beta@f$
 * specialisation; the @f$\nabla\beta@f$ correction term vanishes).
 *
 * Throws @c std::domain_error if @p x coincides with a node — the
 * basis is non-differentiable at conv-hull corners (the dual Hessian
 * degenerates), so the gradient is not well-defined there. Callers
 * doing Galerkin assembly never hit this since quadrature points are
 * strictly interior to integration cells.
 *
 * Parameters and other throws match @ref evaluate_basis.
 */
[[nodiscard]] LMEBasisAndGrad evaluate_basis_and_grad(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 30);

/**
 * @brief Evaluate LME basis values, gradients, **and Hessians** at @p x.
 *
 * Shares the dual Newton solve with @ref evaluate_basis; after Newton
 * converges, the Hessians are computed from the closed-form
 * @f$ \nabla^2 p_a = p_a (v_a v_a^\top - \sum_b p_b \alpha_{ba} v_b
 * v_b^\top - J^{-1}) @f$ — see @ref LMEBasisGradHess for the
 * derivation. The Hessian is essential for biharmonic Galerkin
 * stiffness assembly (the integrand involves
 * @f$ \Delta p_a \Delta p_b @f$ and the mixed second derivatives).
 *
 * Same throws as @ref evaluate_basis_and_grad (input validation,
 * Newton non-convergence, @c std::domain_error at node coincidence).
 */
[[nodiscard]] LMEBasisGradHess evaluate_basis_grad_and_hess(
    const Eigen::MatrixXd&                  nodes,
    const Eigen::VectorXd&                  beta,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 30);

/**
 * @brief One boundary edge of a triangle mesh plus the adjacent
 *        triangle's opposite-corner vertex (the "wing" vertex inside
 *        the mesh).
 *
 * A boundary edge is one that belongs to exactly one triangle. For
 * each such edge @f$(v_0, v_1)@f$ the unique adjacent triangle has a
 * third corner @ref v_int. The triple
 * @f$(v_0, v_1, v_\mathrm{int})@f$ is what the ghost-node
 * construction (Millán 2011 §4.1.2) needs: reflecting
 * @f$v_\mathrm{int}@f$ across the line through @f$v_0 v_1@f$ places
 * a ghost node outside the mesh, edge-symmetric to the interior
 * wing.
 *
 * @see chladni::shell::Edge for the underlying edge representation.
 */
struct BoundaryEdge {
    int v0;     ///< First endpoint, satisfies @c v0 < v1 (per @c Edge).
    int v1;     ///< Second endpoint.
    int v_int;  ///< Third corner of the unique adjacent triangle.
    int face;   ///< Index of the adjacent triangle in @c F.
};

/**
 * @brief Collect every boundary edge of a triangle mesh along with
 *        the adjacent triangle's opposite-corner vertex.
 *
 * Wrapper around @ref chladni::shell::build_edges that filters
 * interior edges and resolves @ref BoundaryEdge::v_int per
 * boundary edge.
 *
 * @param F  Face index matrix, @f$ m \times 3 @f$.
 *
 * @return Vector of @ref BoundaryEdge records, one per boundary edge.
 *         Empty for closed meshes (e.g. icosphere). For a polar disk
 *         the result enumerates the outer rim.
 *
 * @throws std::runtime_error on a non-manifold mesh (propagated from
 *         @ref chladni::shell::build_edges).
 */
[[nodiscard]] std::vector<BoundaryEdge> collect_boundary_edges(
    const Eigen::MatrixXi& F);

/**
 * @brief Reflect a 3D point across the line through two other points.
 *
 * Computes
 * @f$ \mathrm{ghost} = 2\,\mathrm{proj}_{\overline{v_0 v_1}}(v_\mathrm{int})
 *     - v_\mathrm{int} @f$, where
 * @f$ \mathrm{proj}_{\overline{v_0 v_1}}(p) = v_0 + t\,(v_1 - v_0) @f$
 * with @f$ t = \tfrac{(p - v_0) \cdot (v_1 - v_0)}{\|v_1 - v_0\|^2} @f$.
 *
 * The reflection is involutory and preserves distance from the edge
 * line — used to place ghost nodes outside boundary edges so that the
 * LME convex-hull near the boundary extends past the mesh
 * (Millán 2011 §4.1.2; subdivision-FEM-style ghost-node placement,
 * Cirak--Ortiz--Schröder 2000).
 *
 * @throws std::invalid_argument if @c v0 and @c v1 coincide.
 */
[[nodiscard]] Eigen::RowVector3d reflect_across_edge_line(
    const Eigen::RowVector3d& v0,
    const Eigen::RowVector3d& v1,
    const Eigen::RowVector3d& v_int);

/**
 * @brief Build the per-boundary-edge ghost-node position matrix —
 *        a uniform "row of ghost nodes at the boundary"
 *        (Millán 2011 §4.1.2 / RMA13 l.1896).
 *
 * One ghost per boundary edge, placed at the edge MIDPOINT, offset
 * along the in-plane outward normal @f$ n_\mathrm{out} @f$
 * (perpendicular to the edge, pointing away from @c v_int) by the
 * FIRST INTERIOR ROW's directional spacing
 *
 *   @f[ \delta \;=\; \max_{w}\; |(x_w - m)\cdot n_\mathrm{out}|, @f]
 *
 * the maximum over the interior (non-boundary) one-ring neighbours
 * @c w of the edge endpoints — the same max-adjacent directional-
 * spacing rule the SME nodal gaps use (RMA13 §3.2.1). The ghost row
 * thus mirrors the interior sampling row next to the boundary, which
 * is what the papers' reflection construction produces on the
 * uniform-valence rims they show (Fig 8), while staying independent
 * of how the interior happens to be TRIANGULATED.
 *
 * HISTORY (2026-06-03, QuadSplit-rim bug): the previous recipe
 * reflected the adjacent triangle's interior corner @c v_int across
 * the edge line. That inherits triangulation artifacts on rims whose
 * valence is not uniformly 4:
 *  - @c QuadSplit::Checkerboard — adjacent boundary edges share the
 *    same @c v_int, producing near-coincident ghost PAIRS (~0.2 dr
 *    apart) at every other rim vertex;
 *  - @c QuadSplit::UnionJack — @c v_int is the quad-centre vertex at
 *    half the radial spacing, producing a ghost row only ~0.4 dr
 *    outside the rim.
 * Both make ghost basis functions nearly linearly DEPENDENT against
 * their neighbours (node spacing << LME basis width), and the
 * near-dependence directions carry small-but-nonzero K and M — a
 * spurious low-frequency axisymmetric mode below the physical
 * spectrum (32x8 Leissa disk: 45 Hz / 86 Hz vs the physical 234 Hz
 * breathing mode) that no rigid filter can remove. The midpoint +
 * interior-row-offset recipe restores a uniform, well-separated row
 * for ANY rim valence; on uniform valence-4 rims it reproduces the
 * old offsets to within the curved-rim sagitta of the diagonal
 * neighbours (~25% on the 32-gon disk rim — same scale, no
 * near-dependence regime).
 *
 * If an edge has NO interior one-ring neighbour at either endpoint
 * (degenerate strip meshes), @f$ \delta @f$ falls back to the
 * distance of @c v_int from the edge line — the old reflection
 * offset.
 *
 * @param V     Vertex matrix, @f$ N \times 3 @f$.
 * @param F     Face index matrix, @f$ m \times 3 @f$ (source of the
 *              one-ring neighbourhoods).
 * @param bdry  Boundary-edge list from @ref collect_boundary_edges.
 *
 * @return @c (G x 3) ghost-position matrix where @c G == bdry.size().
 *
 * @throws std::invalid_argument if a boundary edge has zero length or
 *         a degenerate (collinear) adjacent triangle.
 */
[[nodiscard]] Eigen::MatrixXd build_ghost_positions(
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::MatrixXi&                   F,
    const std::vector<BoundaryEdge>&         bdry);

/**
 * @brief Classification of a node's position relative to the chart
 *        boundary, controlling its 2nd-order SME nodal-gap matrix.
 *
 * Implements the topological cases of
 * @cite rosolen_millan_arroyo_2013_second_order_maxent §3.2.2 (2D).
 * The Second-order Maximum-Entropy (SME) basis relaxes the canonical
 * 2nd-order moment constraint
 * @f$ \sum_a p_a x_a \otimes x_a = x \otimes x @f$ — which is generally
 * infeasible — to
 *
 *   @f[
 *     \sum_a p_a (x_a - x) \otimes (x_a - x)
 *       = \sum_a p_a\, d_a,
 *   @f]
 *
 * where each per-node symmetric PSD slack matrix @f$ d_a @f$ enlarges
 * the moment-space convex hull just enough to make @f$(x, x \otimes x)@f$
 * feasible. The matrix that a node receives is decided once, at chart
 * construction time, from its local boundary topology — interior nodes
 * get the largest isotropic slack, nodes literally on the boundary get
 * zero, and the in-between cases interpolate (paper Fig. 1).
 */
enum class NodalGapKind : std::uint8_t {
    /// Interior, isotropic spacing.
    /// @f$ d_a = \tfrac{\alpha}{4}\, h_a^2\, I @f$.
    Interior = 0,

    /// Interior, anisotropic spacing.
    /// @f$ d_a = \tfrac{\alpha}{4}\,\sum_{i=1}^2 (h_a^i)^2\, v_a^i \otimes v_a^i @f$
    /// where @f$ (h_a^i, v_a^i) @f$ is the eigendecomposition of the
    /// nodal-spacing metric tensor.
    InteriorAnisotropic = 1,

    /// Corner of the boundary (intersection of two boundary edges).
    /// @f$ d_a = 0 @f$. Same property the 1st-order LME basis already
    /// reproduces — corners interpolate exactly.
    BoundaryCorner = 2,

    /// Interior of a boundary edge AB. With @f$ t @f$ the unit tangent
    /// along AB: @f$ d_a = \tfrac{\alpha}{4}\, h_a^2\, t \otimes t @f$.
    /// Slack only along the boundary, none across it.
    BoundaryEdgeMid = 3,

    /// On a boundary edge, adjacent to a @ref NodalGapKind::BoundaryCorner.
    /// @f$ d_a = \beta\, h_a^2\, t \otimes t @f$ with @f$ \beta \ge 1 @f$.
    /// Slightly enlarged tangential slack to keep feasibility near the
    /// corner pinch.
    BoundaryEdgeNearCorner = 4,

    /// Interior node in the first ring inward from a single boundary
    /// edge, with unit outward normal @f$ n @f$ and tangent @f$ t @f$.
    /// @f$ d_a = \beta\, h_a^2\, n \otimes n + \tfrac{\alpha}{4}\, h_a^2\, t \otimes t @f$.
    NearOneBoundaryEdge = 5,

    /// Interior node in the first ring inward from two boundary edges
    /// (the corner-adjacent interior node), with the two outward unit
    /// normals @f$ n, n' @f$.
    /// @f$ d_a = \beta\, h_a^2\, (n \otimes n + n' \otimes n') @f$.
    NearTwoBoundaryEdges = 6,
};

/**
 * @brief Per-node geometric data for the SME nodal-gap construction.
 *
 * One instance per node, in the local 2D chart coordinate system.
 * Only the fields relevant to the chosen @ref kind are consulted;
 * irrelevant fields are ignored. The defaults (zero vectors, zero
 * scalars) leave the @ref NodalGapKind::Interior kind well-defined as soon as
 * @ref h is set.
 *
 * All unit-vector fields are required to be approximately unit length
 * (@f$ \|\cdot\| \approx 1 @f$ to @c 1e-9) when their @ref kind makes
 * them active; @ref compute_nodal_gaps throws otherwise.
 */
struct NodalGap {
    /// Topological classification — see @ref NodalGapKind.
    NodalGapKind kind = NodalGapKind::Interior;

    /// Characteristic isotropic spacing @f$ h_a @f$. Consulted by every
    /// kind except @ref NodalGapKind::InteriorAnisotropic (which uses
    /// @ref h1 and @ref h2 instead) and @ref NodalGapKind::BoundaryCorner
    /// (which produces @f$ d_a = 0 @f$ unconditionally).
    double h = 0.0;

    /// Unit tangent to the boundary edge passing through (or adjacent to)
    /// this node. Used by @ref NodalGapKind::BoundaryEdgeMid,
    /// @ref NodalGapKind::BoundaryEdgeNearCorner, and
    /// @ref NodalGapKind::NearOneBoundaryEdge.
    Eigen::Vector2d t = Eigen::Vector2d::Zero();

    /// Unit outward normal of the nearest boundary edge. Used by
    /// @ref NodalGapKind::NearOneBoundaryEdge and
    /// @ref NodalGapKind::NearTwoBoundaryEdges (first normal).
    Eigen::Vector2d n = Eigen::Vector2d::Zero();

    /// Second unit outward normal for the two-boundary case. Used by
    /// @ref NodalGapKind::NearTwoBoundaryEdges. May be parallel,
    /// orthogonal, or anything in between to @ref n — corners are
    /// arbitrary-angle in general.
    Eigen::Vector2d n2 = Eigen::Vector2d::Zero();

    /// Directional nodal spacing along the boundary tangent @ref t (RMA13
    /// §3.2.2: the @c t-projected one-ring MAX-adjacent spacing). Used by the
    /// boundary kinds for the @f$ t\otimes t @f$ term. Zero ⇒ fall back to
    /// the isotropic @ref h (e.g. quasi-uniform boundaries, where the
    /// directional and mean spacing coincide).
    double h_t = 0.0;

    /// Directional nodal spacing along the boundary normal @ref n (and
    /// @ref n2). Used by the boundary kinds for the @f$ n\otimes n @f$
    /// term(s). Zero ⇒ fall back to @ref h. On an anisotropic mesh whose
    /// sparse axis is normal to a free edge (e.g. the axially-graded
    /// cylinder), this is the larger axial spacing the @f$ \beta h^2 @f$
    /// normal slack must cover for 2nd-order feasibility.
    double h_n = 0.0;

    /// First eigenvalue of the nodal-spacing metric tensor.
    /// @ref NodalGapKind::InteriorAnisotropic only.
    double h1 = 0.0;

    /// Second eigenvalue of the nodal-spacing metric tensor.
    /// @ref NodalGapKind::InteriorAnisotropic only.
    double h2 = 0.0;

    /// First eigenvector (unit). @ref NodalGapKind::InteriorAnisotropic only.
    Eigen::Vector2d v1 = Eigen::Vector2d::Zero();

    /// Second eigenvector (unit, orthogonal to @ref v1).
    /// @ref NodalGapKind::InteriorAnisotropic only.
    Eigen::Vector2d v2 = Eigen::Vector2d::Zero();
};

/**
 * @brief Compute the per-node SME slack matrices @f$ \{d_a\} @f$.
 *
 * Implements the 2D nodal-gap recipe of
 * @cite rosolen_millan_arroyo_2013_second_order_maxent §3.2.2. Each
 * entry of @p info contributes one symmetric @f$2\times 2@f$ PSD matrix
 * to the output, dispatched on @ref NodalGap::kind —
 *
 *  - @ref NodalGapKind::Interior —
 *      @f$ d_a = \tfrac{\alpha}{4}\, h_a^2\, I @f$.
 *  - @ref NodalGapKind::InteriorAnisotropic —
 *      @f$ d_a = \tfrac{\alpha}{4}\bigl((h_a^1)^2 v_a^1 v_a^{1\top} +
 *                                       (h_a^2)^2 v_a^2 v_a^{2\top}\bigr) @f$.
 *  - @ref NodalGapKind::BoundaryCorner — @f$ d_a = 0 @f$.
 *  - @ref NodalGapKind::BoundaryEdgeMid —
 *      @f$ d_a = \tfrac{\alpha}{4}\, h_a^2\, t\, t^\top @f$.
 *  - @ref NodalGapKind::BoundaryEdgeNearCorner —
 *      @f$ d_a = \beta\, h_a^2\, t\, t^\top @f$.
 *  - @ref NodalGapKind::NearOneBoundaryEdge —
 *      @f$ d_a = \beta\, h_a^2\, n\, n^\top + \tfrac{\alpha}{4}\, h_a^2\, t\, t^\top @f$.
 *  - @ref NodalGapKind::NearTwoBoundaryEdges —
 *      @f$ d_a = \beta\, h_a^2\, (n\, n^\top + n'\, n'^\top) @f$.
 *
 * The output is index-aligned with @p info: result @c [k] is the matrix
 * for node @c k. The matrices are always symmetric; they are PSD by
 * construction (each is a non-negative combination of outer products of
 * real vectors).
 *
 * @param info  Per-node classification + geometry, length @f$N@f$.
 * @param alpha Scalar @f$ \alpha > 1 @f$ from paper §3. Paper recommends
 *              a value @f$ \gtrsim 4 @f$; @f$ \alpha @f$ controls how much
 *              the moment-space convex hull is enlarged in the interior.
 * @param beta  Scalar @f$ \beta \ge 1 @f$ from paper §3. Boundary-layer
 *              enlargement factor. Paper default 1.
 *
 * @return Vector of @f$2 \times 2@f$ symmetric PSD matrices, length
 *         @c info.size().
 *
 * @throws std::invalid_argument if @f$ \alpha \le 1 @f$, @f$ \beta < 1 @f$,
 *         any @c info[k] requires a positive spacing
 *         (@ref NodalGap::h or @ref NodalGap::h1 / @ref NodalGap::h2)
 *         that is non-positive, or any unit-vector field is not unit
 *         to within @c 1e-9 (or for
 *         @ref NodalGapKind::InteriorAnisotropic, @ref NodalGap::v1 and
 *         @ref NodalGap::v2 are not mutually orthogonal to @c 1e-9).
 */
[[nodiscard]] std::vector<Eigen::Matrix2d> compute_nodal_gaps(
    const std::vector<NodalGap>& info,
    double                       alpha,
    double                       beta);

/**
 * @brief Evaluate the 2D 2nd-order SME basis at a single query point.
 *
 * Solves the SME convex-dual program of
 * @cite rosolen_millan_arroyo_2013_second_order_maxent eqs. (5)–(6):
 *
 *   @f[
 *     (\lambda^\star, \mu^\star)(x)
 *       = \arg\min_{(\lambda, \mu) \in \mathbb R^2 \times \mathbb R^{2\times 2}_{\mathrm{symm}}}
 *         \ln Z(x, \lambda, \mu),
 *     \quad
 *     Z(x, \lambda, \mu)
 *       = \sum_a \exp\!\bigl(\lambda \cdot (x - x_a) - \mu : D_a\bigr),
 *   @f]
 *
 *   @f[
 *     D_a = (x_a - x) \otimes (x_a - x) - d_a ,
 *   @f]
 *
 * and returns the SME basis values
 *
 *   @f[
 *     s_a(x)
 *       = \frac{\exp\!\bigl(\lambda^\star \cdot (x - x_a) - \mu^\star : D_a\bigr)}
 *              {Z(x, \lambda^\star, \mu^\star)} .
 *   @f]
 *
 * In contrast to the 1st-order LME basis (@ref evaluate_basis), SME has
 * **no explicit Gaussian prior parameter @f$\beta@f$** — the locality of
 * the basis emerges from @f$\mu^\star@f$ being positive definite, which
 * makes @f$ -\mu^\star : (x - x_a)(x - x_a)^\top @f$ act as a Gaussian
 * decay (paper §3.4). Locality and aspect ratio are instead controlled by
 * the per-node slack matrices @p d via @ref compute_nodal_gaps + the
 * @f$\alpha@f$ parameter.
 *
 * @section properties Properties at convergence
 *
 *  - Partition of unity: @f$ \sum_a s_a = 1 @f$ (always, by construction).
 *  - 1st-order moment: @f$ \sum_a s_a (x - x_a) = 0 @f$ to @p newton_tol.
 *  - 2nd-order moment with gap: @f$ \sum_a s_a D_a = 0 @f$, equivalently
 *    @f$ \sum_a s_a (x_a - x) \otimes (x_a - x) = \sum_a s_a d_a @f$ to
 *    @p newton_tol.
 *  - Nonnegativity: @f$ s_a(x) \ge 0 @f$ for all @c a.
 *  - Weak Kronecker-delta at @f$ d_b = 0 @f$ nodes (boundary corners of
 *    the chart in the SME boundary recipe): @f$ s_a(x_b) = \delta_{ab} @f$.
 *
 * @section newton Newton solver
 *
 * In 2D the dual variable lives in @f$ \mathbb R^5 @f$ — two components
 * for @f$\lambda@f$ plus three independent symmetric entries of
 * @f$\mu@f$. The implementation packs them as
 * @f$ \theta = (\lambda_1, \lambda_2, \mu_{11}, \mu_{22}, \mu_{12}) @f$
 * and per-node augmented offsets
 * @f$ \phi_a = (u_{a,1}, u_{a,2}, -D_{a,11}, -D_{a,22}, -2 D_{a,12}) @f$
 * with @f$ u_a = x - x_a @f$, so that @f$ \theta \cdot \phi_a @f$ equals
 * the log-weight @f$ \lambda \cdot u_a - \mu : D_a @f$. Newton on
 * @f$ \mathcal F(\theta) = \ln Z @f$ then uses
 *
 *   @f[
 *     \nabla \mathcal F = \sum_a s_a\, \phi_a, \quad
 *     \nabla^2 \mathcal F
 *       = \sum_a s_a\, \phi_a \phi_a^\top
 *       - (\nabla\mathcal F)(\nabla\mathcal F)^\top .
 *   @f]
 *
 * Newton from @f$\theta = 0@f$ typically converges in 8–15 iterations on
 * the interior; near the boundary or for irregular node distributions it
 * can need more (paper §3.4.3 Figures 10–11 report up to 25). Per-node
 * exact-match shortcut: if @p x coincides with @f$x_b@f$ within
 * @f$10^{-14}@f$ and @f$ \|d_b\|_F < 10^{-14} @f$, return
 * @f$ s_a = \delta_{ab} @f$ directly. (At interior nodes with @f$ d_b \ne 0 @f$
 * the basis is not delta — Newton runs normally there.)
 *
 * @param nodes
 *   @f$ N \times 2 @f$ row matrix of node positions in the chart.
 *   Only 2D charts are supported in this phase; higher dimensions would
 *   require enlarging the dual to @f$ d + d(d+1)/2 @f$.
 * @param d
 *   Per-node SME slack matrices @f$ d_a @f$, length @c N. Each entry
 *   must be symmetric and positive semi-definite. Typically produced by
 *   @ref compute_nodal_gaps on a per-chart classification of the node
 *   set.
 * @param x
 *   Query point @f$ x \in \mathbb R^2 @f$. Must lie in the moment-space
 *   convex hull of @f$\{(x_a, x_a \otimes x_a - d_a)\}@f$ for the dual
 *   to have a finite optimum (paper §3.1 / §3.2 — slack design); for
 *   queries inside the support that condition is automatically met by
 *   the @ref compute_nodal_gaps construction.
 * @param r_cut
 *   Far-field truncation radius. Nodes with @f$ |x - x_a| > r_{\mathrm{cut}} @f$
 *   are excluded from the active set; this is a numerical cutoff (the
 *   basis is asymptotically Gaussian once @f$\mu^\star@f$ is PD) and
 *   can be set generously without affecting the answer.
 * @param newton_tol
 *   Convergence tolerance on the dual-gradient infinity norm
 *   @f$ \|\nabla \mathcal F\|_\infty @f$. Default @c 1e-10.
 * @param newton_max_iters
 *   Hard upper bound on Newton iterations; @c std::runtime_error on
 *   non-convergence. SME Newton can need more steps than LME — default
 *   @c 50.
 *
 * @return Sparse basis values; see @ref LMEBasisValues.
 *
 * @throws std::invalid_argument on empty @p nodes, mismatched
 *   @c d.size(), @p nodes with @c cols() != 2, @c x.size() != 2, any
 *   asymmetric or negative-eigenvalue @f$d_a@f$, or non-positive
 *   @p r_cut.
 * @throws std::runtime_error if Newton does not converge in
 *   @p newton_max_iters iterations. This typically indicates that the
 *   query point is outside the feasibility region — e.g., all
 *   @f$d_a = 0@f$ (canonical infeasible case from paper §3.1).
 */
[[nodiscard]] LMEBasisValues evaluate_sme_basis(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 50);

/**
 * @brief Evaluate the 2D 2nd-order SME basis **and spatial gradients**
 *        at a single query point.
 *
 * Runs the same dual Newton solve as @ref evaluate_sme_basis; after
 * convergence the per-node gradients are obtained via the implicit
 * function theorem applied to the dual stationarity condition
 * @f$ g(\theta^\star(x), x) = 0 @f$.
 *
 * @section ift Implicit function theorem
 *
 * With @f$ \theta^\star(x) @f$ implicitly defined by
 * @f$ g(\theta, x) = \sum_a p_a(\theta, x)\,\phi_a(x) = 0 @f$,
 * differentiating w.r.t. @f$ x_k @f$ and using @f$ g(\theta^\star) = 0 @f$
 * gives
 *
 *   @f[
 *     H\,\frac{\partial \theta^\star}{\partial x_k} = -G_{,k}, \quad
 *     G_{ik} = \sum_a s_a\,\phi_{a,i}\,r_{a,k}
 *            + \sum_a s_a\,(\partial \phi_{a,i}/\partial x_k),
 *   @f]
 *
 * where @f$ r_{a,k} = \theta^\star \cdot (\partial \phi_a/\partial x_k) @f$
 * and @f$ H = \partial^2 \ln Z / \partial \theta^2 @f$ is the Fisher
 * information already produced by the Newton solve. The per-node
 * gradient is then
 *
 *   @f[
 *     \nabla s_a(x)
 *       = s_a\!\left[
 *           \bigl(\tfrac{\partial \theta^\star}{\partial x}\bigr)^{\!\top}\!\phi_a
 *           \;+\;J_a^\top \theta^\star
 *           \;-\;\sum_b s_b\,J_b^\top \theta^\star
 *         \right] ,
 *   @f]
 *
 * with @f$ J_a = \partial \phi_a/\partial x \in \mathbb R^{5 \times 2} @f$
 * the per-node augmented Jacobian. Implementing this needs no new
 * Newton work; the IFT solve reuses the converged 5×5 Hessian.
 *
 * Consistency identities (used by the unit tests; they hold to Newton
 * tolerance once the dual converges):
 *  - @f$ \sum_a \nabla s_a(x) = 0 @f$ (PoU differentiated).
 *  - @f$ \sum_a x_a \otimes (\nabla s_a)^\top = I @f$ (1st-order
 *    moment differentiated).
 *
 * Parameters and other throws match @ref evaluate_sme_basis. Throws
 * @c std::domain_error if @p x coincides with a node that has
 * @f$ d_b = 0 @f$ (boundary corner) — the basis is non-differentiable
 * there because @ref evaluate_sme_basis returns the delta-Dirac
 * shortcut.
 */
[[nodiscard]] LMEBasisAndGrad evaluate_sme_basis_and_grad(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 50);

/**
 * @brief Evaluate the 2D 2nd-order SME basis, **gradients, and Hessians**
 *        at a single query point.
 *
 * Returns per-node values @f$ s_a(x) @f$, gradients
 * @f$ \nabla s_a(x) \in \mathbb R^2 @f$, and symmetric Hessians
 * @f$ \nabla^2 s_a(x) \in \mathbb R^{2 \times 2} @f$. The gradients
 * are the closed-form IFT result of @ref evaluate_sme_basis_and_grad;
 * the Hessians are produced by central finite differences of those
 * gradients (4 extra Newton + IFT solves per query: @f$ \nabla s_a @f$
 * at @f$ x \pm h e_l @f$ for @f$ l \in \{1, 2\} @f$). The FD step
 * @f$ h = 10^{-5} @f$ gives @f$ \mathcal O(h^2) \approx 10^{-10} @f$
 * accuracy on a smooth interior basis, dominating the Newton
 * tolerance.
 *
 * Closed-form @f$ \nabla^2 s_a @f$ is also derivable from the SME dual
 * (a "second IFT" through the 5×5 Hessian + chain-rule on the
 * gradient closed form) but is non-trivial. The FD path is used here
 * because: (a) the gradient is itself closed-form so the FD layer is
 * shallow, (b) the per-Gauss-point Hessian cost is then 5× that of a
 * basis evaluation, acceptable for Phase B's first-cut biharmonic
 * assembly. A future profiling pass can revisit if needed.
 *
 * Consistency identities (used by the unit tests):
 *  - @f$ \sum_a \nabla^2 s_a(x) = 0 @f$ (Hessian of PoU).
 *  - @f$ \sum_a x_{a,k} \nabla^2 s_a(x) = 0 @f$ for each @f$ k @f$
 *    (Hessian of @f$ \sum s_a x_a = x @f$, which has zero second
 *    derivative in @f$ x @f$).
 *
 * The per-node Hessian is explicitly symmetrised (averaged with its
 * transpose) on output — central FD on the gradient is symmetric only
 * up to the FD truncation residual.
 *
 * **Active-set sensitivity**: because FD evaluates at @f$ x \pm h e_l @f$,
 * the active set could in principle change between adjacent queries
 * if @p r_cut is set near the support boundary. Callers in Galerkin
 * assembly should pick @p r_cut generously (a couple of @f$ h @f$
 * larger than the maximum participating-node distance) so that the
 * active sets at @f$ x \pm h e_l @f$ all match. Mismatched active
 * sets produce silently-incorrect Hessians.
 *
 * Parameters and other throws match @ref evaluate_sme_basis_and_grad.
 * Throws @c std::domain_error if @p x coincides with a corner node
 * (@f$ d_b = 0 @f$, basis non-differentiable).
 */
[[nodiscard]] LMEBasisGradHess evaluate_sme_basis_grad_and_hess(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 50);

/**
 * @brief Closed-form-Hessian variant of @ref evaluate_sme_basis_grad_and_hess.
 *
 * Implements Appendix C of
 * @cite rosolen_millan_arroyo_2013_second_order_maxent :
 * Hessian of @f$ s_a @f$ via the chain rule applied to the SME dual,
 * using a "second IFT" through the 5×5 dual Hessian @f$ J = \partial^2
 * \ln Z / \partial \theta^2 @f$ to get
 * @f$ \partial^2 \theta^\star / \partial x_k \partial x_l @f$, then
 * paper Eq. (C.4) for @f$ \nabla^2 s_a @f$. The per-node formula is
 * derived directly in this module's
 * @f$ \theta = (\lambda_1,\lambda_2,\mu_{11},\mu_{22},\mu_{12}) @f$
 * packing — see @c build_grad_and_hess_from_state in the .cpp for the
 * full transcription.
 *
 * Compared to the FD-on-grad path in @ref evaluate_sme_basis_grad_and_hess
 * this:
 *  - produces the **exact** Hessian rather than a finite difference:
 *    it satisfies the analytic identities @f$ \sum_a \nabla^2 s_a = 0 @f$
 *    and symmetry to machine precision (the FD path reaches ~1e-6),
 *  - runs a single Newton solve per query instead of 1 + 4 warm-started
 *    perturbed solves,
 *  - eliminates the active-set match constraint that pinned SME to a
 *    wide @c r_cut_mult_curved in the assembler.
 *
 * Verified against a Richardson-extrapolated FD reference and the
 * analytic identities via @c [.diag][diag_sme_closed_form]. Not yet
 * wired into the assembler hot path (the FD path remains the default
 * there pending a perf A/B); this entry point is production-ready for
 * callers that want the exact Hessian directly.
 */
[[nodiscard]] LMEBasisGradHess evaluate_sme_basis_grad_and_hess_closed_form(
    const Eigen::MatrixXd&                       nodes,
    const std::vector<Eigen::Matrix2d>&          d,
    const Eigen::Ref<const Eigen::VectorXd>&     x,
    double r_cut,
    double newton_tol       = 1e-10,
    int    newton_max_iters = 50);

}  // namespace lme

/**
 * @brief Local Max-Ent shell assembler (Arroyo--Ortiz 2006 family).
 *
 * Concrete @ref ShellAssembler for the LME meshfree approximant. On a
 * flat plate this ships full Kirchhoff--Love shell physics: bending on
 * the z component (biharmonic Galerkin via the closed-form Hessian of
 * the LME basis) and 2D plane-stress membrane on the @f$(u_x, u_y)@f$
 * components (via the basis gradient). The two are decoupled at zero
 * curvature and combined into the @ref ShellAssembler @f$3 n_V@f$
 * layout. Curved-shell support (Millan 2011 wPCA + Shepard PoU) is
 * implemented and on by default (@c Params::use_curved_shell); only the
 * legacy flat path (@c use_curved_shell = false) requires planar input.
 *
 * @section dof DOF layout
 *
 * One scalar DOF per input vertex per displacement component, matching
 * @ref LoopAssembler — @f$ 3 n_\mathrm{V} @f$ rows/columns. The basis is
 * **non-interpolating** in general (exact interpolation only at
 * conv-hull corners — see weak Kronecker-delta in @ref lme.hpp); the
 * eigenvectors of @f$ K \alpha = \lambda M \alpha @f$ are therefore
 * basis coefficients, not vertex displacements. Conversion to vertex
 * displacements goes through a sparse mat-vec
 * @f$ u_h(x_j) = \sum_a \alpha_a N_a(x_j) @f$, which a future
 * @c evaluate_modes_at_vertices hook on @ref ShellAssembler will
 * dispatch.
 *
 * @section params Parameters
 *
 *  - @c gamma: dimensionless aspect ratio @f$ \gamma @f$ in
 *    @f$ \beta_a = \gamma / h_a^2 @f$. Default @c 0.8 — the thin-shell
 *    value from Millán 2011 Table I / §4.1 (the @c 1.6 in older docs was
 *    that paper's Fig-4 2D *non-shell* demo, corrected by the W-D2
 *    faithfulness fix). Larger = sharper/more local, smaller = wider/
 *    smoother; Rosolen 2013 notes wider support (smaller γ) is better for
 *    bending. GUI slider range @c [0.5, 4.0].
 *  - @c r_cut_mult: truncation radius as a multiple of @f$ h_a @f$.
 *  - @c n_quadrature_per_tri: Galerkin quadrature order on the input
 *    @c F triangles (used once @ref assemble_K is wired).
 *  - @c newton_max_iters, @c newton_tol: dual Newton tolerances; the
 *    same values reach the free @ref lme::evaluate_basis.
 */
class LMEAssembler : public ShellAssembler {
public:
    /**
     * @brief Per-instance configuration of the LME assembler.
     *
     * Defaults track the Millan 2011 / Arroyo--Ortiz 2006 paper
     * recommendations; changing a single field is the intended way to
     * A/B a knob in the GUI tab or a test.
     */
    struct Params {
        /// Dimensionless aspect ratio @f$ \gamma @f$ in
        /// @f$ \beta_a = \gamma / h_a^2 @f$.  Larger = sharper /
        /// near-Delaunay-hat, smaller = wider / smoother. Default 0.8 =
        /// Millán 2011 Table I thin-shell γ_LME; validated on the open
        /// disk (n=2 +0.75 %) and the paper's static shell tests.
        ///
        /// OFF-LABEL CAVEAT (free-free curved membrane shells): the wide
        /// paper-faithful support distorts when a curved neighbourhood is
        /// projected onto the tangent-plane chart, so 1st-order LME locks
        /// hard on the free-free cylinder ovalling modes — a regime the
        /// source papers never test (they use diaphragm ends). Measured
        /// n=2 error vs γ on that cylinder: γ=0.8 +5300 %, γ=2.5 +24 %
        /// (≈ the SME floor), γ=4.0 +97 % — a U-curve minimised near
        /// γ≈2.5. The default stays 0.8 (faithful + disk-optimal); for
        /// free-free curved membrane shells raise γ, or use the SME path
        /// (γ-independent, +23 %) or Loop (no lock, +6 %). See
        /// tests/shell/test_lme_membrane_locking_sri.cpp ([sri_probe]).
        double gamma = 0.8;

        /// Dimensionless aspect ratio @f$ \gamma_{\text{wPCA}} @f$ for the
        /// weighted-PCA tangent-plane fit (Millán 2011 §2.1/§4.1: the
        /// three approximants — wPCA, PU, LME — carry INDEPENDENT aspect
        /// ratios; the paper fixes @f$ \gamma_{\text{wPCA}} = 1.8 @f$). The
        /// chart-frame weight is @f$ \exp(-\gamma_{\text{wPCA}}|P_a-Q_A|^2/h^2)
        /// @f$. Independent of @ref gamma (the LME-basis aspect), which it
        /// previously — and unfaithfully — reused. On a flat mesh the
        /// weighted PCA of coplanar points is @f$ \gamma @f$-independent, so
        /// this only affects curved charts.
        double gamma_wpca = 1.8;

        /// Truncation radius for the basis support on the **flat
        /// path** (@ref use_curved_shell @c = false), as a multiple
        /// of the per-node spacing @f$ h_a @f$. The flat path solves
        /// a single global LME basis at each Gauss point; a tight
        /// cutoff (≤ 2) starves the basis of nodes (no Shepard PoU
        /// to recover coverage) and produces catastrophic accuracy
        /// loss — sweep on SS plate 10x10 (`[.diag][rcut_sweep_ss]`)
        /// at @c r_cut_mult @c = @c 1.32 gives mode 0 at @c -39 %.
        /// 4.0 (≈ all kRing=3 nodes active) is the conservative
        /// default that gives the SS plate its 0.13–0.35 % per-mode
        /// accuracy. Do NOT lower this on the flat path.
        double r_cut_mult = 4.0;

        /// BFS depth used to build each chart's neighbour set on the
        /// curved+ghost path (k-ring through F). Default 3 — measured
        /// to balance chart locality against rigid-kernel preservation
        /// on the representative fixtures. Larger values widen each
        /// chart (more nodes, slower per-Gauss-point Newton, but
        /// fewer chart-boundary effects); too large produces
        /// non-PSD K on high-centre-valence meshes (memory
        /// `[[chladni-stiffness-alternatives]]` §g: kRing=4 on 72x8
        /// polar disk reaches the valence-72 centre at depth 4 and
        /// the wPCA projection on that ~80-node cloud becomes rank-
        /// deficient → silently corrupt basis → 6 negative
        /// eigenvalues at ~5e11 rad²/s². Stay at 3 unless a sweep
        /// proves another value safe.
        int k_ring_depth = 3;

        /// Truncation radius for the per-chart LME basis on the
        /// **curved+ghost path** (@ref use_curved_shell @c = true),
        /// as a multiple of the per-node spacing @f$ h_a @f$.
        /// Independent from @ref r_cut_mult because the curved path
        /// has Shepard PoU recovering coverage across charts.
        ///
        /// ⚠ NUMBERS BELOW ARE PRE-2026-06-03 ERA HISTORY: they were
        /// measured while the closed-form LME derivatives used the
        /// Arroyo-Ortiz 2006 UNIFORM-β special case against per-node
        /// β_a = γ/h_a² (fixed per Millán 2011 App A in `edaf0e1`).
        /// Post-fix the 32x8 polar disk reads +0.75 % at defaults and
        /// the whole tradeoff narrative is moot — this field is a
        /// no-op on the curved LME path anyway (see SUPERSEDED note).
        ///
        /// MESH-DEPENDENT TRADEOFF (re-characterised 2026-05-29 — the
        /// earlier "tighter cutoff = more accurate biharmonic stiffness,
        /// optimum 1.30–1.40" claim was a polar-disk-SPECIFIC result
        /// presented as general; it is misleading). Two effects pull in
        /// opposite directions:
        ///   (1) INTERIOR bending Hessian resolution wants a WIDE stencil.
        ///       On a structured square plate the curved-path bending
        ///       energy is only ~0.64× analytic at 1.4, rising gradually
        ///       to ~0.97× by ~3.0 (no sharp threshold); the Scordelis-Lo
        ///       roof is ~22% too soft at 1.4 and matches the paper
        ///       (~1.02×) at ~4.0. The free-free cylinder is likewise
        ///       bending-starved at 1.4.
        ///   (2) FREE-EDGE fidelity on the polar-disk RADIAL mesh wants a
        ///       NARROW stencil: that mesh already resolves bending at 1.4
        ///       (32x8 free-edge Leissa modal +0.6%), and WIDENING to 4.0
        ///       there regresses the free edge to +7.9% (commit `f16922b`).
        /// So no single value is universally best: the radial disk prefers
        /// ~1.4, structured/cylinder meshes need ~3–4 for correct bending.
        /// The shipped 1.4 was tuned on the disk fixture; it under-resolves
        /// bending on structured/curved meshes — including the GUI's LME
        /// path on such meshes. Closed meshes are insensitive (icosphere
        /// k=2 sweeps within 0.2% across @c [1.2, 6]). See
        /// tests/shell/test_static_obstacle_course.cpp and
        /// chladni-lme-cylinder-locking for the measurements.
        ///
        /// **2nd-order SME wants the OPPOSITE.** With
        /// @ref use_second_order_sme the sharper end hurts: a measured
        /// sweep (2026-05-27) puts the 32x8 polar disk on a ~0.1 %
        /// plateau over @c [2.5, 4.0], degrading to 1.2 % at 1.6 and
        /// DIVERGING at 1.4 (every Shepard-active patch's in-chart
        /// Newton fails). SME's accuracy lever is its slack matrix,
        /// not truncation; SME callers should pin this to ~4.0. The
        /// shipped default stays at 1.4 because LME (the default
        /// approximant) is the common case. SME does NOT read this
        /// field — its truncation is VALUE-based from the paper's
        /// γ_eff = 2/α since 2026-06-04 (inventory C7;
        /// @c r_cut_mult_sme is retired).
        ///
        /// SUPERSEDED 2026-05-29 for the LME curved path: that path now
        /// uses the paper's VALUE-based neighbour cutoff (Millán 2011
        /// Eq. 2) derived from @ref tol_lme, NOT this fixed radius.
        ///
        /// REMOVED 2026-06-10 (review3 N12): the field was retired/inert
        /// since 2026-05-29 (no longer read by any assembler) yet remained
        /// a public, settable member — a silent trap where setting it did
        /// nothing. The historical value was 1.4; the live knob is now
        /// @ref tol_lme.
        // (r_cut_mult_curved removed — see @ref tol_lme.)

        /// Neighbour-search cutoff TOLERANCE for the 1st-order LME basis
        /// on the curved path — the paper's value-based truncation
        /// (Millán 2011 Eq. 2 / Table I, @f$ TOL_{LME} = 10^{-10} @f$):
        /// node @f$ a @f$ enters the active set iff
        /// @f$ \exp(-\beta_a |x-x_a|^2) > TOL_{LME} @f$, i.e. within the
        /// per-node radius @f$ h_a\sqrt{\ln(1/TOL_{LME})/\gamma} @f$
        /// (@f$ \beta_a = \gamma/h_a^2 @f$). This is mesh-ADAPTIVE: the
        /// support scales with local spacing @f$ h_a @f$ and needs no
        /// per-mesh tuning, replacing the fixed @c r_cut_mult_curved
        /// (whose narrow 1.4 starved the bending Hessian on structured/
        /// curved meshes — see test_static_obstacle_course.cpp). At the
        /// shipped @ref gamma the cutoff radius (~3.8 h) exceeds the
        /// k-ring patch extent, so the patch's full node set is used and
        /// bending is resolved correctly; closed/supported fixtures match
        /// the paper (Scordelis-Lo ~1.02×). Free-edge plate modes land
        /// sub-1 % with ghosts on (32x8 polar disk Leissa n=2 +0.75 %).
        /// (An earlier "~8 % genuine 1st-order floor" claim here was
        /// FALSIFIED 2026-06-03: it was the uniform-β closed-form
        /// derivative bug, fixed per Millán 2011 App A in `edaf0e1`.)
        double tol_lme = 1.0e-10;

        /// Neighbour-search cutoff tolerance used to size the **chart
        /// node set** (the patch's wPCA + in-chart Newton support) on
        /// the 1st-order LME curved path — DELIBERATELY looser than
        /// @ref tol_lme. The chart extent is
        /// @f$ h_a\sqrt{\ln(1/\mathtt{chart\_tol\_lme})/\gamma} @f$;
        /// at the paper's @f$ \gamma_{LME}=0.8 @f$ the basis cutoff
        /// @ref tol_lme @f$ =10^{-10} @f$ would give a 5.36 h chart
        /// (~103 nodes on a uniform mesh, an O(n²)-per-Gauss-point
        /// Newton hot spot — the curved disk timing regressed ~6× when
        /// the geometric support landed). Sizing the CHART from
        /// @f$ 10^{-6} @f$ (= the paper's @f$ TOL_{PU} @f$) shrinks it
        /// to 4.16 h (~62 nodes, ~2.8× faster); the nodes between the
        /// two radii carry basis weight @f$ <10^{-6} @f$, so dropping
        /// them from the chart is numerically negligible. Only consulted
        /// on the 1st-order LME path (SME keeps its k-ring chart).
        double chart_tol_lme = 1.0e-6;

        /// Hard cap on the number of nodes in a single LME chart on the
        /// curved path. Bounds the in-chart Newton cost (O(n_act²)) and,
        /// crucially, prevents a FREEZE at very-high-valence vertices
        /// (e.g. a polar disk's valence-144 centre, which the geometric
        /// support reaches from many patches → 145+ chart nodes). When a
        /// geometrically-selected chart exceeds the cap, only the nearest
        /// @ref max_chart_nodes (by Euclidean distance to the anchor) are
        /// kept — the dropped nodes are the farthest, hence lowest-weight,
        /// so well-shaped meshes (chart ≪ cap) are unaffected. Set high
        /// enough that it never bites on quality meshes (~96).
        int max_chart_nodes = 96;

        /// SUPERSEDED 2026-06-04 (C7 closed): the SME truncation
        /// radius is now VALUE-BASED, mirroring the LME path with the
        /// paper's own effective decay rate — RMA13 §3.4.3 derives
        /// @f$ \mu \approx (2/\alpha)/h^2 @f$, "the same way that
        /// @f$ \beta_a = \gamma/h_a^2 @f$ does for LME", so the
        /// assemblers use @f$ \gamma_{\mathrm{eff}} = 2/\alpha @f$ in
        /// the Eq. 2 cutoff:
        /// @f$ r = h\sqrt{\ln(1/TOL_{LME})\,\alpha/2} @f$ (≈ 4.80 h
        /// at the shipped α=2, on the tuned constant's measured
        /// @c [2.5, 4.0] plateau — the k-ring chart bounds the active
        /// set anyway, so the radius only must not CLIP inside the
        /// chart; the old 1.4 default did, and diverged).
        ///
        /// REMOVED 2026-06-10 (review3 N12): retired/inert since
        /// 2026-06-04 (the SME truncation is value-based,
        /// @f$ \gamma_{\mathrm{eff}} = 2/\alpha @f$) yet remained a
        /// public, settable member — a silent trap. The historical value
        /// was 4.0.
        // (r_cut_mult_sme removed — SME truncation is value-based.)

        /// EXPERIMENTAL (chart-unification A/B rounds 1–3,
        /// 2026-06-04): when @c > 0, the 2nd-order SME path builds
        /// its chart with the INTRINSIC (geodesic) extractor
        /// (@ref lme::extract_chart_neighbourhood_intrinsic, Dijkstra
        /// over edge lengths, radius @c sme_chart_radius_mult · h_a)
        /// instead of the legacy flat topological k-ring
        /// (@ref k_ring_depth). Round-3 winner of the A/B: the
        /// isotropic extrinsic ball (round 1) read +349 % on the
        /// aspect-2.39 cylinder (arc-folding) and the directional
        /// max-projection variant (round 2,
        /// @ref lme::extract_chart_neighbourhood_directional) lost
        /// the cylinder ladder at every level; the intrinsic form is
        /// the only one to beat the k-ring at the coarsest
        /// anisotropic point (+36.0 vs +45.6 %) but pays at finer
        /// levels and is non-monotonic in refinement — measured
        /// 2026-06-06 to be locking-mediated chart-shape sensitivity
        /// (NOT a gap-classification artifact; see the scaffolding
        /// inventory C4 refutation note). Default 0 = k-ring, the
        /// shipped configuration.
        double sme_chart_radius_mult = 0.0;

        /// Bounded per-patch α escalation on SME in-chart failure
        /// (2nd-order path only). When a patch's in-chart SME solve
        /// fails at a quadrature point — marginal 2nd-order
        /// infeasibility, e.g. the coarse geodesic pinched hemisphere
        /// at the paper's α=2 once the faithful 12-pt quadrature
        /// (Q-D2) probes near triangle corners — the evaluation is
        /// retried with the patch's gap matrices enlarged ×2 per step
        /// (@f$ \alpha_{\mathrm{eff}} = \alpha\,2^k @f$, up to this
        /// many steps) before falling through to the drop-patch
        /// safety net. Gap ENLARGEMENT is the paper's own feasibility
        /// mechanism (@cite rosolen_millan_arroyo_2013_second_order_maxent
        /// §3.2–3.3) and α its free regularisation parameter — which
        /// the paper itself already grades spatially near boundaries
        /// (α interior vs β boundary), so a locally enlarged α at a
        /// handful of marginal quadrature points is in-family, and
        /// strictly gentler than the alternative of dropping the
        /// patch from the PoU entirely. 0 disables (failure goes
        /// straight to the drop net). LME_DIAG reports recoveries.
        int sme_alpha_escalation_steps = 2;

        /// Abort threshold for the in-chart-Newton drop net. After the
        /// per-patch ownership filter (paper-faithful, routine) AND the
        /// α-escalation ladder have run, any patch whose Newton solve still
        /// diverges is dropped from a Gauss point's partition of unity and
        /// the survivors are renormalised. If the dropped Newton-failure
        /// weight at a single quadrature point exceeds this fraction of the
        /// point's PoU, assembly throws instead of silently folding a
        /// badly-perturbed block into K/M. This guards the
        /// "silent degradation" failure mode: on all quality meshes the
        /// Newton channel is dormant (drop weight 0), so the default never
        /// fires; it only trips on a genuine local breakdown. The ROUTINE
        /// ownership-filter drops are excluded from this budget (they are
        /// the §4.1.1 per-patch quadrature mechanism, not failures).
        /// >= 1.0 effectively disables the abort (renormalise-anything,
        /// the legacy behaviour).
        double max_newton_drop_frac = 0.25;

        /// Abort threshold for the TOTAL partition-of-unity loss at a
        /// quadrature point — the actual renormalisation denominator
        /// @f$ 1 - w_{\mathrm{failed}} @f$, which sums BOTH the routine
        /// §4.1.1 ownership-filter drops AND the Newton-failure drops.
        /// @c max_newton_drop_frac budgets only the Newton channel; the
        /// ownership channel was previously bounded only by the implicit
        /// @f$ w_{\mathrm{failed}} < 1 @f$ guard, so a Gauss point could
        /// shed an arbitrarily large fraction of its PoU through the
        /// ownership filter and still be folded into K/M after a large
        /// rescale (review3 N2). This caps the combined loss so the
        /// renormalisation can never silently absorb a near-total PoU
        /// shed. Default 0.9 leaves generous headroom above the worst
        /// MEASURED legitimate cumulative drop (≈0.52 on the coarse
        /// lat-long hemisphere) while still surfacing the pathological
        /// "≈1.0 → rescale blows up" case. >= 1.0 disables it.
        double max_total_drop_frac = 0.9;

        /// Galerkin quadrature rule on input-mesh triangles, by sample
        /// count: 1 (@c OnePointCentroid), 3 (@c ThreePointEdgeMid),
        /// 7 (@c SevenPointDunavant, degree-5) or 12
        /// (@c TwelvePointDunavant, degree-6). Default 12 — the
        /// paper's rule (Millán 2011 §4.1.1: "a standard
        /// Gauss–Legendre cubature rule of 12 points (order 6) per
        /// triangle"; wired as Q-D2 of the 2026-06-03 faithfulness
        /// audit). Lower orders are for [.diag] quadrature-sensitivity
        /// probes; both `assemble_K_curved_bending` and
        /// `assemble_M_curved` consume this field.
        int n_quadrature_per_tri = 12;

        /// Hard upper bound on dual Newton iterations.
        int newton_max_iters = 20;

        /// Newton convergence tolerance on
        /// @f$ \|\nabla_\lambda \ln Z\|_\infty @f$.
        double newton_tol = 1e-10;

        /// Select the curved-shell assembly path (Millán 2011 §2: wPCA
        /// patches + Shepard PoU + per-chart 2D LME). Default @c true
        /// since 2026-05-25 (commit pinning `use_curved_shell` flip)
        /// — the curved path accepts both flat and curved meshes, and
        /// since the @c r_cut_mult_curved split landed it also gives
        /// SIGNIFICANTLY better accuracy than the flat path on the
        /// representative open-boundary fixtures (32x8 polar disk
        /// Leissa: flat-path +14.9 %, curved+ghost +0.6 %). Setting
        /// to @c false reverts to the legacy flat-plate path
        /// (Arroyo-Ortiz 2006 single global basis) which is faster but
        /// only handles strictly-planar meshes and has the worse
        /// accuracy on free-edge fixtures.
        bool use_curved_shell = true;

        /// Aspect-ratio parameter for the Shepard partition-of-unity
        /// over patch anchors. Paper default range
        /// @f$ \gamma_{\mathrm{PU}} \in [3, 6] @f$
        /// (@cite millan_rosolen_arroyo_2011_thin_shell_maxent Table I).
        /// Only consulted when @c use_curved_shell is @c true; per-patch
        /// Shepard rate is @f$ \beta_A = \gamma_{\mathrm{PU}} / h_A^2 @f$.
        double gamma_pu = 4.0;

        /// Augment the LME node set with ghost nodes outside boundary
        /// edges (Millán 2011 §4.1.2; subdivision-FEM placement à la
        /// Cirak–Ortiz–Schröder 2000). One ghost per boundary edge,
        /// reflected across the edge line from the opposite triangle
        /// corner; the ghost participates in patch neighbour lists
        /// near the boundary but does not host its own Shepard PoU
        /// patch. Structurally avoids most of the chart-convex-hull
        /// truncation at the rim (residual cases on larger meshes are
        /// caught by the in-worker drop-and-renormalise inside the
        /// curved-shell assemblers).
        ///
        /// When @c true, @ref assemble_K and @ref assemble_M (curved
        /// path only) return matrices of size @f$3(N + G)@f$ where
        /// @f$G@f$ is the boundary-edge count; the trailing @f$G@f$
        /// rows/cols hold ghost DOFs that callers must handle (slice
        /// off displayed mode shapes; keep in the eigenvalue solve).
        bool use_ghost_nodes = true;

        /// Use the 2nd-order Symmetric Maximum-Entropy basis
        /// (@cite rosolen_millan_arroyo_2013_second_order_maxent)
        /// in place of the 1st-order Local Max-Ent basis at per-
        /// Gauss-point evaluation in the curved-shell K and M
        /// assembly paths. Off by default to preserve the current
        /// performance characteristics; opt in for membrane-dominated
        /// curved fixtures where 1st-order LME locks (free-free
        /// cylinder: LME +274 % vs SME +46 %; Scordelis-Lo SME
        /// ~1.002×). On bending-dominated fixtures the two are now
        /// equivalent — 32x8 polar-disk Leissa n=2: LME +0.75 %,
        /// SME +0.73 %. (The "~8 % 1st-order floor" that originally
        /// motivated SME was FALSIFIED 2026-06-03: it was the
        /// uniform-β derivative bug, fixed in @c edaf0e1.)
        ///
        /// SME relaxes the canonical 2nd-order moment constraint
        /// to be feasible (paper §3) via per-node symmetric PSD
        /// "gap" matrices @f$ d_a @f$ produced by
        /// @ref lme::compute_nodal_gaps from the chart-local
        /// boundary topology of the patch's node set. Locality
        /// is controlled by @ref sme_alpha (interior nodes) and
        /// @ref sme_beta (near-boundary nodes); the LME locality
        /// parameter @ref gamma (i.e. @f$\beta_a@f$) is ignored
        /// on the SME path — SME has no Gaussian prior, the
        /// per-node @f$ \mu^\star @f$ provides locality internally
        /// (paper §3.4).
        ///
        /// Requires @ref use_curved_shell @c true (the SME basis is
        /// 2D, evaluated on the per-patch wPCA chart); throws
        /// @c std::invalid_argument from @ref assemble_K /
        /// @ref assemble_M otherwise.
        bool use_second_order_sme = false;

        /// SME interior-slack scalar @f$ \alpha > 1 @f$ in
        /// @f$ d_a = (\alpha/4) h_a^2 I @f$ for interior nodes.
        /// Paper @cite rosolen_millan_arroyo_2013_second_order_maxent
        /// §3.2 / §4 operates in @f$ \alpha \in [1.6, 2.5] @f$; default
        /// 2.0 (the paper's central value). Larger @f$ \alpha @f$ widens
        /// the basis support and, on curved shells, INCREASES bending
        /// over-stiffness (locking) — Scordelis-Lo runs 1.01×ref at
        /// @f$ \alpha=2 @f$ vs 1.10× at 4; the closed sphere 4.9 % vs
        /// 11 %. The earlier 4.0 default was a workaround for a Newton
        /// convergence-criterion bug that mis-reported feasible @f$
        /// \alpha=2 @f$ solves as divergences (fixed 2026-06); it is no
        /// longer needed. Only consulted when @ref use_second_order_sme.
        double sme_alpha = 2.0;

        /// SME boundary-layer scalar @f$ \beta \ge 1 @f$ in
        /// @f$ d_a = \beta h_a^2 n \otimes n + \ldots @f$ for
        /// near-boundary nodes. Paper default 1.0. Only consulted
        /// when @ref use_second_order_sme is on.
        double sme_beta = 1.0;
    };

    /// Construct from explicit @ref Params. Validates the PoU drop-fraction
    /// budgets (@ref Params::max_newton_drop_frac,
    /// @ref Params::max_total_drop_frac are non-negative and not NaN) and the
    /// chart-sizing parameters that feed the @c sqrt(-log(·)) cutoff-radius
    /// expressions in the curved path — these run before any
    /// @ref lme::evaluate_basis call, so its per-call boundary checks do not
    /// cover them: @ref Params::gamma, @ref Params::gamma_wpca,
    /// @ref Params::gamma_pu and @ref Params::r_cut_mult `> 0`,
    /// @ref Params::tol_lme and @ref Params::chart_tol_lme in `(0, 1)`, and —
    /// only when @ref Params::use_second_order_sme — @ref Params::sme_alpha
    /// `> 1` and @ref Params::sme_beta `>= 1` (their governing consumer
    /// @ref lme::compute_nodal_gaps requires those stricter bounds). Remaining
    /// input checks happen at the per-call boundary in @ref lme::evaluate_basis.
    /// @throws std::invalid_argument on a negative/NaN drop-fraction or an
    ///   out-of-range chart-sizing parameter.
    explicit LMEAssembler(Params params);

    /// Default-constructed: @c LMEAssembler{Params{}}.
    LMEAssembler() : LMEAssembler(Params{}) {}

    /// Read-only access to the configuration.
    [[nodiscard]] const Params& params() const noexcept { return params_; }

    /**
     * @brief Assemble the @f$3 n_V \times 3 n_V@f$ stiffness matrix.
     *
     * Direct sum of two decoupled flat-plate contributions:
     *
     *  - **Bending** on the z component: scalar biharmonic Galerkin
     *    @f$ \int D\,[\Delta p_a \Delta p_b - (1-\nu)(\cdots)]\,dA @f$,
     *    with @f$ D = @f$@c material.k_B. Lands on @f$K[3a+2, 3b+2]@f$.
     *  - **Membrane** on @f$(u_x, u_y)@f$: 2D plane-stress
     *    @f$ \int k_L\,\varepsilon^T D_{\mathrm{mem}}\,\varepsilon\,dA @f$
     *    with engineering shear (Belytschko 1994), driven by
     *    @c material.k_L. Lands on @f$K[3a+i, 3b+j]@f$ for
     *    @f$i, j \in \{0, 1\}@f$.
     *
     * The kernel is the 6-dim rigid-body subspace exactly — three
     * translations plus three rotations of @f$\mathbb R^3@f$. Either
     * @c k_L or @c k_B may be zero (disables that contribution); throws
     * @c std::invalid_argument if both are zero. Curved shells are
     * implemented and on by default (@ref Params::use_curved_shell); only
     * the legacy flat branch throws on non-flat input.
     */
    [[nodiscard]] Eigen::SparseMatrix<double> assemble_K(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const ShellMaterial&   material) const override;

    /**
     * @brief Assemble the @f$3 n_V \times 3 n_V@f$ consistent mass matrix.
     *
     * @f$ M_{ab} = \int \rho h\, p_a(x)\, p_b(x)\,dA @f$ on the input
     * mesh, evaluated with 7-point Dunavant quadrature on each triangle.
     * The three displacement components share a single scalar block;
     * there is no cross-component coupling (Kirchhoff plate decouples
     * in-plane and out-of-plane at the mass level). Curved shells are
     * implemented and on by default (@ref Params::use_curved_shell); only
     * the legacy flat branch throws on non-flat input.
     */
    [[nodiscard]] Eigen::SparseMatrix<double> assemble_M(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        double                 surface_density) const override;

    /**
     * @brief Drop trailing ghost-DOF rows when @ref Params::use_ghost_nodes
     *        is on.
     *
     * The ghost-extended K and M are sized @f$3(N + G)@f$; the eigensolver
     * returns eigenvectors of the same height. For displayed mode shapes
     * we want only the @f$3 N@f$ real-vertex rows. The slice is exact in
     * the LME weak Kronecker-delta sense (real-vertex coefficients are
     * approximately the nodal displacement amplitudes at those
     * vertices); ghost DOFs encode the basis-extension support past the
     * boundary and have no physical-vertex interpretation.
     *
     * When @c use_ghost_nodes is off this is a pass-through (base class
     * behavior).
     */
    [[nodiscard]] Eigen::MatrixXd evaluate_modes_at_vertices(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& mode_coefficients) const override;

    /// Short label used by the GUI tab bar.
    [[nodiscard]] std::string label() const override;

    /**
     * @brief PoU-drop statistics of the most recent curved
     *        @ref assemble_K call on this assembler instance.
     *
     * The B1 watchdog (scaffolding inventory), made queryable: the two
     * silent drop channels of the curved-K partition-of-unity — the
     * §4.1.1 quadrature ownership filter and the in-chart Newton
     * failure net — tallied unconditionally (the counters cost nothing
     * off the drop paths; the verbose per-anchor breakdown stays
     * behind LME_DIAG). @ref DropStats::n_newton must be ZERO on quality meshes
     * now that the SME quadrature ownership is INTERIOR (every Gauss
     * query keeps a full 1-ring of chart-node surround, the certified
     * sufficient condition from the 2026-06-07 rim-infeasibility
     * dissection); a regression gate pins this on the free-free
     * cylinder. Reset at the start of each curved assemble_K.
     */
    struct DropStats {
        long   n_own       = 0;  ///< ownership exclusions (by design)
        long   n_newton    = 0;  ///< in-chart Newton failures (B1 net)
        long   n_escal_ok  = 0;  ///< failures rescued by α escalation
        double w_own       = 0.0;  ///< cumulative PoU weight, ownership
        double w_newton    = 0.0;  ///< cumulative PoU weight, B1 net
        double w_gauss_max = 0.0;  ///< worst single-Gauss dropped weight
    };

    /// @copydoc DropStats
    [[nodiscard]] const DropStats& last_drop_stats() const
    {
        return last_drops_;
    }

private:
    Params params_;
    /// Filled by the curved @ref assemble_K (see @ref DropStats).
    mutable DropStats last_drops_;
};

}  // namespace chladni::shell
