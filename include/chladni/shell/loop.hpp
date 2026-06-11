#pragma once

/**
 * @file loop.hpp
 * @brief Loop subdivision-surface shell FEM (Cirak-Ortiz-Schroder 2000).
 *
 * Strict @f$C^1@f$ thin-shell finite-element analysis using Loop's
 * approximating subdivision scheme. This module is the planned
 * replacement for the CST membrane + Wardetzky-IBM bending K assembly
 * in @ref chladni::shell. The motivation is documented in commit
 * 6b45830 and the project memory: CST + IBM are direction-dependent
 * on regular triangulations and produce 1.5–2x over-stiffness against
 * analytical free-edge thin-plate references because the underlying
 * @f$C^0@f$ piecewise-linear basis cannot represent the second
 * derivatives that thin-plate bending energy needs. Loop subdivision
 * shells solve this by elevating the basis to @f$C^1@f$ without adding
 * rotational DOFs.
 *
 * @section overview Overview
 *
 * Each control-mesh triangle defines an element of the limit surface.
 * The displacement field within an element depends on the 1-ring of
 * vertices around the triangle:
 *  - **Regular patches** (all 3 corner valences == 6): exactly 12
 *    box-spline control points. Closed-form quartic basis functions
 *    in barycentric coordinates @f$ (u, v, w) @f$ from Cirak-Ortiz
 *    Eq. (75) / Stam Appendix A.
 *  - **Irregular patches** (one corner valence @f$N \neq 6@f$): @f$N+6@f$
 *    control points. Stam 1999 eigendecomposition of the subdivision
 *    matrix gives exact basis-function evaluation at any @f$(v, w)@f$
 *    without explicit subdivision iterations.
 *  - **Boundary patches** (any corner on the mesh boundary): the
 *    1-ring is augmented with phantom (ghost) vertices using
 *    Schweitzer's two-temporary-vertex method (Cirak-Ortiz Eq. 54).
 *    Phantom-vertex displacements are linear constraints on real-vertex
 *    displacements (or zero for clamped boundary conditions).
 *
 * Following Cirak-Ortiz Section 4.6 step 1, the algorithm assumes
 * meshes in which every triangle has at most one irregular vertex.
 * Pre-subdivision to enforce that condition is the responsibility of
 * a separate (planned) routine.
 *
 * @section loop_refs References
 * - @cite cirak_ortiz_schroder_2000_subdivision_shells — the foundational
 *   paper. Section 3 (FEM discretisation), Section 4 (subdivision basis),
 *   Section 5 (numerical examples), Appendix A (basis functions and
 *   subdivision matrix block forms). Cited equations: (37) plane-stress
 *   H matrix, (43-45) K assembly, (54) boundary phantom rule, (75)
 *   regular box-spline basis, (79-80) M^I and B^I strain matrices.
 * - @cite stam_1999_loop_evaluation — eigendecomposition machinery for
 *   irregular-patch evaluation. Section 3.2 (eigenstructure), Section 4
 *   (EvalSurf algorithm), Appendix B (subdivision matrix S, S_12,
 *   S_21, S_22 explicit forms), Appendix C (N=3 Jordan-block case).
 *
 * @section loop_status Implementation status
 *
 * The full pipeline is shipped: patch-stencil enumeration,
 * limit-surface evaluation (including the Stam exact eigenbasis for
 * extraordinary patches), boundary phantom vertices, per-element
 * @f$K@f$/@f$M@f$ kernels, and global assembly
 * (@ref chladni::shell::loop::assemble_stiffness_loop /
 * @ref chladni::shell::loop::assemble_mass_loop).
 */

#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>  // QuadratureRule

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <vector>

namespace chladni::shell::loop {

/**
 * @brief Compute the valence (number of incident edges) of every vertex.
 *
 * On a manifold triangle mesh each interior vertex has valence equal to
 * the number of incident triangles (since interior vertices see a
 * closed fan of @f$N@f$ faces with @f$N@f$ edges). On the boundary,
 * valence equals the number of incident faces plus one (open fan).
 *
 * The "regular vs irregular" classification of Cirak-Ortiz Sec. 4.2
 * uses interior valence: a vertex is *regular* iff its valence equals
 * @f$6@f$, and *irregular* (or *extraordinary*) otherwise. Boundary
 * vertices are handled separately (Schweitzer phantom rules) and the
 * regularity criterion does not apply directly to them.
 *
 * @param n_vertices  Number of vertices in the mesh (@c V.rows()).
 * @param edges       Edge list returned by @ref chladni::shell::build_edges.
 *
 * @return Length-@f$n@f$ vector of integer valences.
 */
[[nodiscard]] std::vector<int> vertex_valences(
    Eigen::Index n_vertices,
    const std::vector<Edge>& edges);

/**
 * @brief Per-triangle 1-ring patch DOF stencil.
 *
 * Captures the vertex IDs that participate in the displacement field
 * within one triangle, plus the metadata needed to choose the right
 * basis-evaluation path:
 *  - regular box-spline (Cirak-Ortiz Eq. 75) when @ref is_regular,
 *  - Stam eigenanalysis when irregular interior,
 *  - Schweitzer phantom-vertex augmentation when @ref has_boundary.
 *
 * The "1-ring" of a triangle is the union of the triangles that share
 * at least one vertex with it. The patch DOF set is the set of vertices
 * appearing in those triangles. For an interior regular patch this is
 * exactly 12 vertices (3 corners + 9 ring); for an interior irregular
 * patch with one corner of valence @f$N \neq 6@f$ it is @f$N + 6@f$
 * vertices (assuming pre-subdivision so at most one corner is irregular).
 *
 * Storage convention: @ref corners holds the 3 triangle vertices
 * verbatim from @c F.row(@ref tri_index). @ref ring holds the
 * remaining 1-ring vertex indices, sorted ascending for determinism.
 * Canonical Cirak-Ortiz Fig. 9 ordering (corners at slots 4, 7, 8 of a
 * 12-vector) is applied separately at limit-surface evaluation time
 * and is not encoded in this struct.
 */
struct PatchStencil {
    /// Triangle row index in @c F.
    Eigen::Index                 tri_index;
    /// Triangle corner vertex indices, verbatim from @c F.row(tri_index).
    std::array<Eigen::Index, 3>  corners;
    /// Vertex valences of the 3 corners, in the same order as @ref corners.
    std::array<int, 3>           corner_valences;
    /// 1-ring vertices excluding the 3 corners, sorted ascending.
    std::vector<Eigen::Index>    ring;
    /// True iff at least one corner or ring vertex lies on a mesh boundary edge.
    bool                         has_boundary;

    /**
     * @brief Total number of vertex DOFs in this patch.
     *
     * Equals @c 3 (corners) + @c ring.size(). For regular patches this
     * is exactly @c 12. For irregular interior patches with one
     * valence-@f$N@f$ corner this is @f$N + 6@f$. Boundary patches
     * without phantom-vertex augmentation may have a smaller count;
     * the augmentation is performed separately by L.5.
     */
    [[nodiscard]] Eigen::Index n_dofs() const noexcept
    {
        return 3 + static_cast<Eigen::Index>(ring.size());
    }

    /**
     * @brief True iff every corner has valence 6 and no patch vertex is on the boundary.
     *
     * Regular patches use the closed-form 12-vertex box-spline basis
     * (Cirak-Ortiz Eq. 75) without any subdivision step. Irregular
     * patches require Stam 1999 eigenanalysis; boundary patches
     * require Schweitzer's phantom-vertex augmentation.
     */
    [[nodiscard]] bool is_regular() const noexcept
    {
        return !has_boundary
            && corner_valences[0] == 6
            && corner_valences[1] == 6
            && corner_valences[2] == 6;
    }

    /**
     * @brief Highest valence among the 3 corner vertices.
     *
     * For irregular interior patches with at most one extraordinary
     * vertex (i.e. a pre-subdivided mesh), the patch DOF count is
     * @f$\max(\text{corner valences}) + 6@f$. This helper makes that
     * arithmetic easy to express.
     */
    [[nodiscard]] int max_corner_valence() const noexcept
    {
        return std::max({corner_valences[0],
                         corner_valences[1],
                         corner_valences[2]});
    }
};

/**
 * @brief Enumerate 1-ring patch stencils for every triangle in the mesh.
 *
 * For each triangle this routine:
 *  -# Records the 3 corner vertex indices verbatim from @c F.row(f).
 *  -# Computes the corner valences from the edge list.
 *  -# Walks the 1-ring (the union of triangles sharing any corner with
 *     @c f, minus @c f itself) to build the @ref PatchStencil::ring
 *     vertex list.
 *  -# Marks @ref PatchStencil::has_boundary if any corner or ring
 *     vertex sits on a mesh boundary edge.
 *
 * @ref PatchStencil::ring is sorted ascending so the output is
 * deterministic and easy to compare in tests. Reordering into the
 * canonical Cirak-Ortiz Fig. 9 layout (corners at slots 4, 7, 8 of a
 * 12-vector) is performed at limit-surface evaluation time and is
 * not part of this routine.
 *
 * @param V  Vertex matrix, @f$n \times 3@f$ (only @c V.rows() is read).
 * @param F  Face index matrix, @f$m \times 3@f$.
 *
 * @return Length-@f$m@f$ vector of patch stencils, one per triangle,
 *         in face-row order.
 *
 * @throws std::runtime_error
 *         on a non-manifold edge — same condition as
 *         @ref chladni::shell::build_edges.
 */
[[nodiscard]] std::vector<PatchStencil> build_patch_stencils(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

// ---------------------------------------------------------------------------
// Regular box-spline basis (Cirak-Ortiz Eq. 75 / Stam 1999 Appendix A).
//
// On a regular patch (every triangle corner has valence 6) the limit
// surface is exactly a quartic box-spline with 12 control points laid
// out per Cirak-Ortiz Fig. 9. The 12 basis functions are explicit
// polynomials in the barycentric coordinates @f$(u, v, w)@f$ with
// @f$ u + v + w = 1 @f$; the local curvilinear coordinates of the
// element are @f$(\theta^1, \theta^2) = (v, w)@f$.
//
// Properties (verified by tests):
//  * Partition of unity: @f$ \sum_{I=1}^{12} N_I(v, w) = 1 @f$ everywhere
//    on the unit triangle.
//  * At the centroid @f$(v, w) = (1/3, 1/3)@f$, the three "triangle-
//    corner" basis values @f$N_4 = N_7 = N_8 = 23/81@f$ by 3-fold
//    symmetry of the hexagonal stencil.
//  * The basis is degree 4: each @f$N_I@f$ is a sum of degree-4
//    monomials in @f$(u, v, w)@f$, so @f$\partial^3 N_I / \partial v^a
//    \partial w^b = 0@f$ for @f$a + b = 3@f$ on average; second
//    derivatives are linear in @f$(v, w)@f$.
//
// Slot convention: @c N(0) is @f$N_1@f$ in Cirak-Ortiz Fig. 9
// numbering (the 1-indexed paper numbering minus one). The 3 triangle
// corners sit at slots 3, 6, 7 (i.e. @f$N_4, N_7, N_8@f$).
// ---------------------------------------------------------------------------

/**
 * @brief Evaluate the 12 regular box-spline basis functions at @f$(v, w)@f$.
 *
 * Closed-form quartic polynomials per @cite cirak_ortiz_schroder_2000_subdivision_shells
 * Eq. (75) and @cite stam_1999_loop_evaluation Appendix A. The third
 * barycentric coordinate is @f$ u = 1 - v - w @f$; valid input domain
 * is @f$ v \in [0, 1],\, w \in [0, 1 - v] @f$ (the standard unit triangle).
 *
 * @param v  First curvilinear coordinate, @f$\theta^1 \in [0, 1]@f$.
 * @param w  Second curvilinear coordinate, @f$\theta^2 \in [0, 1 - v]@f$.
 *
 * @return Length-12 vector @f$(N_1, \ldots, N_{12})^\top@f$ of basis
 *         values in Cirak-Ortiz Fig. 9 ordering (slot 0 = @f$N_1@f$).
 *         Sums to 1 exactly within floating-point round-off.
 */
[[nodiscard]] Eigen::Matrix<double, 12, 1> regular_basis(double v, double w);

/**
 * @brief First partial derivatives of the regular basis at @f$(v, w)@f$.
 *
 * Returns @f$ \partial N_I / \partial v @f$ and
 * @f$ \partial N_I / \partial w @f$ for @f$I = 1, \ldots, 12@f$, with
 * @f$ u = 1 - v - w @f$ treated as a function of @f$(v, w)@f$.
 *
 * Implemented analytically by differentiating the closed-form quartic
 * polynomials of @ref regular_basis. Verified by tests against central
 * finite differences of the basis values themselves and against the
 * partition-of-unity invariant
 * @f$ \sum_I \partial N_I/\partial v = \sum_I \partial N_I/\partial w = 0 @f$.
 *
 * @param v  First curvilinear coordinate.
 * @param w  Second curvilinear coordinate.
 *
 * @return @f$12 \times 2@f$ matrix with column 0 = @f$\partial N/\partial v@f$
 *         and column 1 = @f$\partial N/\partial w@f$.
 */
[[nodiscard]] Eigen::Matrix<double, 12, 2> regular_basis_grad(double v, double w);

/**
 * @brief Second partial derivatives of the regular basis at @f$(v, w)@f$.
 *
 * Returns the three independent components of the Hessian of each
 * basis function with respect to @f$(v, w)@f$:
 *  - column 0: @f$\partial^2 N_I/\partial v^2@f$
 *  - column 1: @f$\partial^2 N_I/\partial v \partial w@f$
 *  - column 2: @f$\partial^2 N_I/\partial w^2@f$
 *
 * Implemented analytically with the same chain-rule machinery as
 * @ref regular_basis_grad. Verified against central second differences
 * of basis values and against partition-of-unity invariants
 * @f$ \sum_I \partial^2 N_I/\partial v \partial w = 0 @f$ etc.
 *
 * @param v  First curvilinear coordinate.
 * @param w  Second curvilinear coordinate.
 *
 * @return @f$12 \times 3@f$ matrix of second-derivative components.
 */
[[nodiscard]] Eigen::Matrix<double, 12, 3> regular_basis_hess(double v, double w);

/**
 * @brief Map a regular interior patch's 1-ring to the canonical Cirak-Ortiz
 *        Fig. 9 slot ordering.
 *
 * Returns the 12 vertex indices of the patch arranged so that slot @c i
 * (0-indexed) corresponds to control point @f$N_{i+1}@f$ in
 * @cite cirak_ortiz_schroder_2000_subdivision_shells Eq. (75) and
 * @cite stam_1999_loop_evaluation Fig. 1. In the 1-indexed paper
 * convention the three triangle corners sit at slots 4, 7, 8; here
 * (0-indexed) those become slots 3, 6, 7.
 *
 * Algorithm: walk CCW around each of the 3 corners, starting from the
 * "next" corner in the @c F.row winding. The CCW spokes give exactly
 * the 6-vertex 1-ring of the corner; combined across the 3 corners
 * (with shared vertices identified by the topology) the 12 slots fill
 * in canonically. The mapping per walk is:
 *  - spoke 0 = next corner
 *  - spoke 1 = previous corner (= next-next)
 *  - spoke 2 = edge-opposite of edge to previous corner
 *  - spokes 3-4 = the 2 outer-ring vertices of this corner
 *  - spoke 5 = edge-opposite of edge to next corner
 *
 * The convention places spoke 3 (the first outer-ring vertex CCW after
 * the edge-opposite-of-prev) into the slot adjacent to that edge-
 * opposite vertex in the Fig. 9 layout, and spoke 4 into the slot
 * adjacent to the next edge-opposite vertex. See the source for the
 * explicit slot table.
 *
 * @param stencil  Patch stencil whose 3 corners must all have
 *                 valence 6 with closed (interior) fans.
 * @param F        Face index matrix (must be the same @c F passed to
 *                 @ref build_patch_stencils).
 *
 * @return 12 vertex indices in canonical Fig. 9 slot ordering.
 *
 * @throws std::invalid_argument
 *         if any corner's valence is not 6.
 * @throws std::runtime_error
 *         if any corner has an open fan (boundary vertex), so that the
 *         CCW walk cannot wrap around. Use the L.5 phantom-vertex path
 *         for boundary patches instead.
 */
[[nodiscard]] std::array<Eigen::Index, 12> canonical_regular_dofs(
    const PatchStencil& stencil,
    const Eigen::MatrixXi& F);

/**
 * @brief Gather the @f$ K = N + 6 @f$ canonical Stam-patch DOFs for an
 *        irregular interior patch with exactly one extraordinary corner.
 *
 * Mirrors @ref canonical_regular_dofs for the irregular case. The patch
 * must have exactly one corner of valence @f$ N \ne 6 @f$ (the
 * extraordinary vertex); the other two corners must have valence 6 with
 * closed (interior) fans. This is the standard Cirak-Ortiz / Stam
 * configuration after one Loop subdivision step: original irregular
 * vertices are isolated by valence-6 edge-midpoints, so every irregular
 * face has exactly one extraordinary corner.
 *
 * Output layout matches @c build_irregular_patch in loop_stam.cpp:
 *  - slot @c 0           : extraordinary vertex @c ev (valence @f$ N @f$).
 *  - slots @c 1..N       : CCW 1-ring of @c ev starting from the "next"
 *                          corner @c v_1 of the face (i.e. the corner
 *                          following @c ev in @c F.row winding).
 *  - slot @c N+1         : vertex across edge @c (v_1, v_N) from @c ev.
 *  - slot @c N+2         : vertex across edge @c (v_1, N+1) from @c v_1
 *                          (the "next" CCW neighbour of @c v_1 after
 *                          slot @c N+1; in the canonical patch this is
 *                          one of @c ev's original 1-ring neighbours).
 *  - slot @c N+3         : vertex across edge @c (v_1, v_2) from @c ev.
 *  - slot @c N+4         : vertex across edge @c (v_N, N+1) from @c v_N
 *                          (mirror of slot @c N+2).
 *  - slot @c N+5         : vertex across edge @c (v_{N-1}, v_N) from @c ev.
 *
 * The slot layout exactly reproduces the canonical patch built by
 * @c build_irregular_patch (loop_stam.cpp) — see the diagrams there.
 *
 * Algorithm: each "across edge from X" lookup uses the directed-edge
 * map built once for @p F. The CCW 1-ring walk around @c ev is the same
 * machinery as @ref canonical_regular_dofs.
 *
 * @param stencil  Patch stencil. Exactly one corner must have valence
 *                 @f$ N \ne 6 @f$; the other two must have valence 6.
 * @param F        Face index matrix (same one passed to
 *                 @ref build_patch_stencils).
 *
 * @return Length-@f$ K = N + 6 @f$ vertex-index vector in the canonical
 *         Stam-patch slot ordering described above.
 *
 * @throws std::invalid_argument
 *         if zero or more than one corner has valence @f$ \ne 6 @f$, or
 *         if the regular corners have valence @f$ \ne 6 @f$.
 * @throws std::runtime_error
 *         if any walk hits an open fan (boundary vertex), or if the
 *         CCW 1-ring of @c ev does not close back to @c v_1 after @f$ N @f$
 *         steps (non-manifold or inconsistent topology).
 */
[[nodiscard]] std::vector<Eigen::Index> gather_stam_patch_dofs(
    const PatchStencil& stencil,
    const Eigen::MatrixXi& F);

/**
 * @brief Limit-surface quantities evaluated at one parametric point.
 *
 * Holds everything the Cirak-Ortiz strain matrices need: the 3D
 * position, the covariant tangent basis @f$a_\alpha@f$, the surface
 * normal @f$a_3@f$, and the parametric second derivatives
 * @f$x_{,\alpha\beta}@f$ — together with the underlying basis
 * function values, gradient, and Hessian (kept around so callers
 * assembling per-element K do not need to recompute them).
 *
 * @section frame Surface frame
 *
 * Following @cite cirak_ortiz_schroder_2000_subdivision_shells Eq. (3),
 * (8), (9):
 *  - @ref cov_basis columns are the parametric tangent vectors
 *    @f$a_1 = x_{,v}, a_2 = x_{,w}@f$, both 3D.
 *  - @ref normal is the unit director
 *    @f$a_3 = (a_1 \times a_2) / \|a_1 \times a_2\|@f$, well-defined
 *    iff the parametric jacobian @f$|a_1 \times a_2|@f$ is non-zero.
 *  - @ref second_derivs columns are the parametric second derivatives
 *    @f$a_{1,1} = x_{,vv}, a_{1,2} = a_{2,1} = x_{,vw}, a_{2,2} = x_{,ww}@f$,
 *    each a 3D vector.
 */
struct PatchEvaluation {
    /// 3D position @f$x(v, w)@f$ of the limit surface.
    Eigen::Vector3d                position;
    /// 3 x 2 covariant tangent basis: column 0 = @f$x_{,v}@f$, column 1 = @f$x_{,w}@f$.
    Eigen::Matrix<double, 3, 2>    cov_basis;
    /// Unit surface normal @f$a_3 = (x_{,v} \times x_{,w}) / \|\cdot\|@f$.
    Eigen::Vector3d                normal;
    /// 3 x 3 parametric second derivatives: columns
    /// @f$x_{,vv}, x_{,vw}, x_{,ww}@f$.
    Eigen::Matrix<double, 3, 3>    second_derivs;
    /// Basis values @f$N_I(v, w)@f$ in canonical Fig. 9 slot order.
    Eigen::Matrix<double, 12, 1>   N;
    /// Basis gradient: column 0 = @f$\partial N/\partial v@f$, column 1 = @f$\partial N/\partial w@f$.
    Eigen::Matrix<double, 12, 2>   N_grad;
    /// Basis Hessian: columns @f$\partial^2 N/\partial v^2, \partial^2 N/\partial v\partial w, \partial^2 N/\partial w^2@f$.
    Eigen::Matrix<double, 12, 3>   N_hess;
};

/**
 * @brief Evaluate the regular Loop limit surface at @f$(v, w)@f$.
 *
 * Gathers the 12 control vertices given by @p canonical_dofs from
 * @p V into a 12 x 3 matrix @f$P@f$ and contracts against the basis
 * to produce @ref PatchEvaluation. Concretely:
 *  - position      = @f$P^\top N(v, w)@f$
 *  - cov_basis     = @f$P^\top \nabla N(v, w)@f$
 *  - second_derivs = @f$P^\top \nabla^2 N(v, w)@f$
 *  - normal        = @f$(a_1 \times a_2) / \|\cdot\|@f$
 *
 * The basis values, gradient, and Hessian themselves are returned in
 * the @c N, @c N_grad, @c N_hess members so that L.4's per-element K
 * routine can build @f$M^I@f$ and @f$B^I@f$ matrices without re-
 * evaluating the basis.
 *
 * For one-point quadrature at the centroid of the element, call with
 * @c v = @c w = 1/3 (Cirak-Ortiz Section 4.6).
 *
 * @param canonical_dofs  Output of @ref canonical_regular_dofs for a
 *                        regular interior patch.
 * @param V               Vertex matrix, @f$n \times 3@f$, in the same
 *                        configuration whose limit surface is wanted.
 * @param v, w            Parametric evaluation point on the standard
 *                        unit triangle, @f$v, w \geq 0@f$ and
 *                        @f$v + w \leq 1@f$.
 *
 * @return A fully populated @ref PatchEvaluation.
 *
 * @throws std::invalid_argument if @c V has fewer than 3 columns.
 */
[[nodiscard]] PatchEvaluation evaluate_patch_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&               V,
    double                               v,
    double                               w);

/**
 * @brief Per-element stiffness matrix for a regular interior Loop patch.
 *
 * Implements @cite cirak_ortiz_schroder_2000_subdivision_shells Eq. (43),
 * integrated by the @p rule quadrature (default @ref
 * QuadratureRule::SevenPointDunavant, degree-5; the original Cirak-Ortiz
 * Sec. 4.6 one-point centroid rule and the other @ref quadrature_points
 * rules are also accepted). Per Gauss point @f$g@f$ with weight
 * @f$w_g@f$,
 * @f[
 *   K_e \;=\; \sum_g w_g \sqrt{\bar a}\,\bigl[
 *      \tfrac{E h}{1-\nu^2}\, M^\top H M
 *    + \tfrac{E h^3}{12(1-\nu^2)}\, B^\top H B
 *   \bigr],
 * @f]
 * where @f$M, B \in \mathbb{R}^{3 \times 36}@f$ are the Voigt strain-
 * displacement matrices stacked over the 12 patch DOFs, and
 * @f$H \in \mathbb{R}^{3 \times 3}@f$ is the plane-stress isotropic
 * elasticity in contravariant-metric form (Eq. 37).
 *
 * Per-DOF strain blocks come from Appendix A.0.3:
 *  - @f$M^I@f$ rows from Eq. (79): @f$N^I_{,1} a_1@f$, @f$N^I_{,2} a_2@f$,
 *    @f$N^I_{,2} a_1 + N^I_{,1} a_2@f$ — the third row already gives
 *    @f$2\alpha_{12}@f$ by combining the symmetric contributions, so
 *    no additional Voigt factor is needed.
 *  - @f$B^I@f$ rows from Eq. (80) for @f$B^I_1, B^I_2, B^I_3@f$. The
 *    third row is multiplied by 2 because Eq. (80) gives the scalar
 *    @f$\beta_{12}@f$ while the Voigt convention requires
 *    @f$2\beta_{12}@f$ (consistent with @f$\alpha@f$'s third Voigt
 *    component).
 *
 * The basis values, gradient, and Hessian needed for @f$M^I@f$ and
 * @f$B^I@f$ are evaluated at each quadrature point via
 * @ref evaluate_patch_regular on the **rest** geometry.
 *
 * @param canonical_dofs  Output of @ref canonical_regular_dofs (the 12
 *                  patch DOFs in Fig. 9 slot order).
 * @param V_aug     Augmented rest-vertex matrix (same as for K assembly).
 * @param material  Membrane / bending prefactors and Poisson ratio.
 * @param rule      Reference-triangle quadrature rule (default 7-point
 *                  Dunavant degree-5).
 *
 * @return @f$36 \times 36@f$ symmetric element stiffness matrix; rows
 *         and columns are blocks of 3 (one per spatial component), 12
 *         such blocks (one per DOF in canonical Fig. 9 slot order).
 *
 * @throws std::runtime_error
 *         on a degenerate parametrisation
 *         (@f$|a_1 \times a_2| = 0@f$).
 */
[[nodiscard]] Eigen::Matrix<double, 36, 36> element_stiffness_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&              V_aug,
    const ShellMaterial&                material,
    QuadratureRule                      rule = QuadratureRule::SevenPointDunavant);

/**
 * @brief Per-element consistent mass matrix for a regular interior Loop patch.
 *
 * Computes
 * @f[
 *   M_e^{IJ} \;=\; \rho h \int_{T} N_I(v, w)\,N_J(v, w)\,\sqrt{a(v, w)}\,dv\,dw,
 * @f]
 * with @f$\sqrt{a} = |a_1 \times a_2|@f$ the area element on the limit
 * surface and the integral taken over the standard unit triangle @f$T@f$.
 * Quadrature is 7-point Dunavant degree 5 — exact for polynomials of
 * total degree @f$\le 5@f$; the quartic Loop basis gives @f$N_I N_J@f$
 * of degree 8, so the residual under-integration is @f$O(h^6)@f$ per
 * element, well below the FEM truncation error.
 *
 * The 36 x 36 element matrix is block-diagonal across the three spatial
 * components: each (I, J) block is @f$M_e^{IJ} \cdot I_3@f$.
 *
 * @param canonical_dofs  Output of @ref canonical_regular_dofs.
 * @param V_aug           Augmented vertex matrix (same as for K assembly).
 * @param surface_density @f$\rho h@f$ (kg/m²), @f$> 0@f$.
 * @param rule            Reference-triangle quadrature rule (default
 *                        7-point Dunavant degree-5).
 *
 * @return Symmetric positive-semi-definite element mass @f$\in \mathbb{R}^{36 \times 36}@f$.
 *
 * @throws std::runtime_error on a degenerate parametrisation at any
 *         quadrature point.
 */
[[nodiscard]] Eigen::Matrix<double, 36, 36> element_mass_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&              V_aug,
    double                              surface_density,
    QuadratureRule                      rule = QuadratureRule::SevenPointDunavant);

// ---------------------------------------------------------------------------
// Boundary handling: Schweitzer phantom-vertex augmentation
// (Cirak-Ortiz Eq. 54 / Section 4.5).
//
// Loop subdivision needs a closed valence-6 fan around every patch
// corner so that the regular box-spline basis applies. Boundary
// vertices have open fans and so their fans need to be closed by
// adding "phantom" vertices outside the mesh. Schweitzer's rule places
// one phantom across each boundary edge, computed by linearly
// extrapolating the third vertex of the unique adjacent face:
// @f[
//     p_e \;=\; v_0 + v_1 - v_\text{int},
// @f]
// where @f$\{v_0, v_1\}@f$ are the boundary edge endpoints and
// @f$v_\text{int}@f$ is the third vertex of the unique boundary face.
// The constraint is linear so the phantom contributes no new degrees
// of freedom: each phantom DOF is a fixed combination of three real
// DOFs, captured in a sparse constraint matrix @f$C@f$.
//
// **Valence-4 boundary vertices** (cylinder rims, interior strip
// edges): Schweitzer's rule alone closes the fan to valence 6:
// 2 real boundary edges + 2 real interior edges + 2 phantom edges.
// One gap-closing triangle (v, p_in, p_out) per corner.
//
// **Valence-3 boundary vertices** (rectangular plate corners after
// the diagonal-flip topology of @ref chladni::mesh::generate_flat_plate):
// the two across-edge phantoms close the fan only to valence 5 because
// a valence-3 corner has just one interior neighbour, not two. One
// extra "corner phantom" closes the gap. Position is the parallelogram
// rule
// @f[
//     p_\text{corner} \;=\; p_\text{in} + p_\text{out} - v
//         \;=\; v + b_\text{in} + b_\text{out} - 2 u,
// @f]
// where @f$ b_\text{in}, b_\text{out} @f$ are the two boundary
// neighbours and @f$ u @f$ is the (unique) interior neighbour. The
// gap is filled with two corner-fill triangles
// @f$ (v, p_\text{in}, p_\text{corner}) @f$ and
// @f$ (v, p_\text{corner}, p_\text{out}) @f$. Sum of constraint
// coefficients is 1 so the corner phantom translates rigidly with the
// input.
//
// **Valence-2 boundary corners** (single-triangle corners — strip
// endpoints, raw rectangular plate corners): two corner phantoms via
// the reflection rule
// @f[
//     p_A \;=\; 2 v - b_\text{out},\qquad
//     p_B \;=\; 2 v - b_\text{in},
// @f]
// each constraint row summing to 1. The gap is filled with three
// corner-fill triangles
// @f$ (v, p_\text{in}, p_A), (v, p_A, p_B), (v, p_B, p_\text{out}) @f$
// — three wedges of the open boundary angle. Unblocks the strip vs
// Euler-Bernoulli sibling test and lets a non-chamfered rectangular
// plate be assembled directly.
// ---------------------------------------------------------------------------

/**
 * @brief Augmented mesh + linear constraint matrix for boundary patches.
 *
 * The augmentation appends @ref n_phantom phantom vertices to @c V and
 * @ref n_phantom_faces phantom triangles to @c F so that every original
 * boundary corner has a closed valence-6 fan in the augmented mesh.
 * Phantoms are NOT independent DOFs; their displacements are linear
 * combinations of real DOFs, encoded in @ref C.
 *
 * Layout (with @f$n_\text{phantom\_edge}@f$ across-edge phantoms — one
 * per boundary edge — and @f$n_\text{phantom\_corner}@f$ corner
 * phantoms — @f$(4 - \text{valence})@f$ per boundary corner: 0 for
 * valence-4 (closed by the across-edge phantoms alone), 1 for valence-3,
 * 2 for valence-2):
 *  - @c V_aug.topRows(n_real) is exactly @c V.
 *  - @c V_aug.middleRows(n_real, n_phantom_edge) holds the across-edge
 *    phantoms in @ref chladni::shell::build_edges' boundary-edge order.
 *  - @c V_aug.bottomRows(n_phantom_corner) holds the corner phantoms
 *    in ascending boundary-vertex order.
 *  - @c F_aug.topRows(n_real_faces) is exactly @c F.
 *  - @c F_aug.bottomRows(n_phantom_faces) holds the phantom triangles:
 *    one across each boundary edge, plus @f$(5 - \text{valence})@f$
 *    gap-fill triangles per boundary corner (1 for valence-4, 2 for
 *    valence-3, 3 for valence-2).
 *  - @c C is sparse with shape @f$3 n_\text{aug} \times 3 n_\text{real}@f$
 *    where @f$n_\text{aug} = n_\text{real} + n_\text{phantom}@f$. The
 *    top @c 3*n_real rows form the identity (real DOFs map to themselves);
 *    each across-edge-phantom row has 3 nonzeros @f$(+1, +1, -1)@f$ on
 *    @f$(v_0, v_1, v_\text{int})@f$. Corner-phantom rows depend on
 *    valence: a valence-3 corner contributes one row with 4 nonzeros
 *    @f$(+1, +1, +1, -2)@f$ on @f$(v, b_\text{in}, b_\text{out}, u)@f$
 *    (parallelogram rule); a valence-2 corner contributes two rows via
 *    the reflection rule @f$p = 2 v - b@f$, each with 2 nonzeros
 *    @f$(+2, -1)@f$ on @f$(v, b_\text{out})@f$ and @f$(v, b_\text{in})@f$
 *    respectively. Every row sums to 1 so phantoms translate rigidly.
 *
 * The reduced stiffness matrix is @f$K = C^\top K_\text{aug} C@f$ where
 * @f$K_\text{aug}@f$ is assembled over the original real triangles only
 * (the phantom triangles are connectivity-only and do not contribute
 * material energy).
 */
struct LoopAugmented {
    /// Augmented vertex matrix, @f$(n_\text{real} + n_\text{phantom}) \times 3@f$.
    Eigen::MatrixXd              V_aug;
    /// Augmented face matrix, @f$(n_\text{real\_faces} + n_\text{phantom\_faces}) \times 3@f$.
    Eigen::MatrixXi              F_aug;
    /// Sparse constraint @f$C \in \mathbb R^{3 n_\text{aug} \times 3 n_\text{real}}@f$.
    Eigen::SparseMatrix<double>  C;
    /// Number of real (input) vertices.
    Eigen::Index                 n_real;
    /// Number of real (input) faces.
    Eigen::Index                 n_real_faces;
    /// Number of phantom vertices appended to @c V.
    Eigen::Index                 n_phantom;
    /// Number of phantom triangles appended to @c F.
    Eigen::Index                 n_phantom_faces;
};

/**
 * @brief Augment a triangle mesh with Schweitzer phantom vertices for
 *        Loop subdivision boundary handling.
 *
 * Implements @cite cirak_ortiz_schroder_2000_subdivision_shells Eq. (54).
 * For every boundary edge a phantom vertex is added at the position
 * @f$v_0 + v_1 - v_\text{int}@f$. For every boundary vertex a
 * gap-closing triangle is added connecting the two phantoms across its
 * two boundary edges. After augmentation every original boundary
 * vertex has valence 6 in the augmented mesh and a closed CCW fan
 * suitable for @ref canonical_regular_dofs.
 *
 * **Phantom triangle winding.** For each boundary edge with real face
 * containing directed edge @f$v_0 \to v_1@f$, the phantom triangle has
 * winding @f$(v_1, v_0, p)@f$ so that the augmented mesh's
 * directed-edge-to-face map is 1-to-1. For each boundary vertex
 * @f$v@f$, the gap-closing triangle has winding
 * @f$(v, p_\text{in}, p_\text{out})@f$ where @f$p_\text{in}@f$ is the
 * phantom across the boundary edge whose real-face directed edge
 * points into @f$v@f$, and @f$p_\text{out}@f$ is the phantom across
 * the edge whose real-face directed edge points out of @f$v@f$.
 *
 * @param V  Input vertex matrix, @f$n_\text{real} \times 3@f$.
 * @param F  Input face matrix, @f$n_\text{real\_faces} \times 3@f$.
 *
 * @return Populated @ref LoopAugmented.
 *
 * @throws std::invalid_argument
 *         if any boundary vertex has valence other than 2, 3, or 4.
 * @throws std::runtime_error
 *         if any boundary vertex has more than 2 boundary edges
 *         (non-manifold boundary), or if the local corner topology
 *         is inconsistent with its valence (also a non-manifold case).
 */
[[nodiscard]] LoopAugmented augment_for_loop_boundary(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

// ---------------------------------------------------------------------------
// One global Loop subdivision step (Cirak-Ortiz Section 4.6 step 1).
//
// Loop subdivision refines each input triangle into 4 sub-triangles by
// adding one new vertex per edge and re-triangulating. The new vertex
// positions are linear combinations of the original control vertices
// captured by Loop's masks:
//
//  * Interior odd rule (new edge midpoint, @f$\{v_0, v_1\}@f$ shared by
//    two faces with opposite vertices @f$c_L, c_R@f$):
//    @f[ p_e = \tfrac{3}{8}(v_0 + v_1) + \tfrac{1}{8}(c_L + c_R) @f]
//  * Boundary odd rule (new edge midpoint on a single-face edge):
//    @f[ p_e = \tfrac{1}{2}(v_0 + v_1) @f]
//  * Interior even rule (smoothed original vertex of valence @f$n@f$):
//    @f[ p_v = (1 - n\beta_n) v + \beta_n \sum_{i=1}^{n} v_i,\quad
//        \beta_n = \tfrac{1}{n}\bigl(\tfrac{5}{8} -
//        (\tfrac{3}{8} + \tfrac{1}{4}\cos(2\pi/n))^2\bigr),\quad
//        \beta_3 = \tfrac{3}{16} @f]
//  * Boundary even rule (smoothed boundary vertex with two boundary
//    neighbours @f$v_p, v_n@f$):
//    @f[ p_v = \tfrac{3}{4} v + \tfrac{1}{8}(v_p + v_n) @f]
//
// All four rules are linear in the input positions, so the entire
// subdivision step is captured by a single sparse @f$3 n_\text{sub}
// \times 3 n_\text{real}@f$ matrix @f$S@f$ with
// @f$ n_\text{sub} = n_\text{real} + n_\text{edges} @f$. Subdivision
// preserves valence at original-vertex slots; new boundary edge
// midpoints have valence 4 in the subdivided mesh, new interior edge
// midpoints have valence 6 (regular). Original extraordinary vertices
// remain extraordinary with the same valence in the subdivided mesh.
//
// In the L.3.4 irregular-patch path this routine is composed with
// @ref augment_for_loop_boundary: subdivide once to push the irregular
// area down to one residual sub-triangle per original irregular
// triangle (1/4 of original area), then augment phantom boundary
// vertices, assemble on the subdivided + augmented mesh, and reduce
// back to original DOFs via @f$ K = S^\top C^\top K_\text{aug} C\,S @f$.
// The residual still-irregular sub-triangles are skipped on the
// subdivided mesh (Cirak-Ortiz Section 4.6 step 1 approximation; the
// missing energy is bounded by the irregular sub-triangle area which
// shrinks geometrically with each additional subdivision pass).
// ---------------------------------------------------------------------------

/**
 * @brief Subdivided mesh + linear constraint matrix from one Loop step.
 *
 * Fields:
 *  - @ref V_sub holds @f$ n_\text{real} + n_\text{edges} @f$ rows: the
 *    first @ref n_real are the smoothed positions of the input
 *    vertices (Loop's even rule); rows
 *    @c [n_real, n_real + n_edge_midpoints) are the new edge-midpoint
 *    vertices in the same order as @ref chladni::shell::build_edges.
 *  - @ref F_sub holds @f$ 4 n_\text{real\_faces} @f$ sub-triangles. For
 *    input face @c F.row(f) = (a, b, c) with edge midpoints @f$ e_{ab},
 *    e_{bc}, e_{ca} @f$, the four sub-triangles at rows @c 4f..4f+3 are
 *    laid out as:
 *      - @c 4f+0: @c (a, e_ab, e_ca)        — corner sub-triangle near a
 *      - @c 4f+1: @c (e_ab, b, e_bc)        — corner sub-triangle near b
 *      - @c 4f+2: @c (e_ca, e_bc, c)        — corner sub-triangle near c
 *      - @c 4f+3: @c (e_ab, e_bc, e_ca)     — central (medial) sub-triangle
 *    All four preserve the CCW winding of the parent.
 *  - @ref S has shape
 *    @f$ 3(n_\text{real}+n_\text{edges}) \times 3 n_\text{real} @f$.
 *    Each spatial component (x, y, z) is constrained independently by
 *    the same scalar weight, giving 3 nonzeros per nonzero entry of the
 *    underlying scalar mask. Multiplying @c S by a stacked
 *    @c [V.col(0); V.col(1); V.col(2)] vector reproduces the stacked
 *    @ref V_sub. (For convenience @ref V_sub itself is materialised at
 *    the same time so callers do not have to apply @ref S.)
 */
struct LoopSubdivision {
    /// Subdivided vertex positions, @f$(n_\text{real}+n_\text{edges}) \times 3@f$.
    Eigen::MatrixXd              V_sub;
    /// Subdivided face indices, @f$4 n_\text{real\_faces} \times 3@f$.
    Eigen::MatrixXi              F_sub;
    /// Sparse linear constraint @f$ S \in \mathbb R^{3 n_\text{sub} \times 3 n_\text{real}}@f$.
    Eigen::SparseMatrix<double>  S;
    /// Number of input vertices.
    Eigen::Index                 n_real;
    /// Number of input faces.
    Eigen::Index                 n_real_faces;
    /// Number of new edge-midpoint vertices (= number of edges in input).
    /// Only meaningful as the @c [n_real, n_real + n_edge_midpoints) layout
    /// offset after a SINGLE subdivision pass; @ref loop_subdivide_n_times
    /// sets it to @c -1 for @c n_passes > 1, where V_sub interleaves several
    /// subdivision generations and the simple offset no longer applies.
    Eigen::Index                 n_edge_midpoints;
};

/**
 * @brief Apply one Loop subdivision step to a triangle mesh.
 *
 * Implements the four Loop masks (interior odd, boundary odd, interior
 * even, boundary even) and packages the result with the linear
 * constraint matrix @f$ S @f$ that captures the subdivision as a
 * sparse linear map on stacked DOF displacements.
 *
 * Smoothing weights:
 *  - Interior valence-@f$n@f$ vertex:
 *    @f$ \beta_3 = 3/16 @f$;
 *    @f$ \beta_n = (1/n)\bigl(5/8 - (3/8 + (1/4)\cos(2\pi/n))^2\bigr) @f$
 *    for @f$n \neq 3@f$;
 *    centre weight @f$ 1 - n\beta_n @f$, neighbour weight @f$\beta_n@f$.
 *  - Boundary vertex with the two boundary neighbours @f$v_p, v_n@f$:
 *    centre weight @f$3/4@f$; the two boundary-neighbour weights
 *    @f$1/8@f$ each. Interior neighbours of a boundary vertex do
 *    **not** contribute to the boundary even rule.
 *
 * Edge midpoint weights (odd rule):
 *  - Interior edge: @f$ 3/8 @f$ on each endpoint, @f$ 1/8 @f$ on the
 *    third vertex of each adjacent face.
 *  - Boundary edge: @f$ 1/2 @f$ on each endpoint.
 *
 * @param V  Input vertex matrix, @f$n_\text{real} \times 3@f$.
 * @param F  Input face matrix, @f$n_\text{real\_faces} \times 3@f$. CCW
 *           winding (matching the rest of the @ref chladni::shell API).
 *
 * @return Populated @ref LoopSubdivision.
 *
 * @throws std::runtime_error
 *         if a boundary vertex has more or fewer than 2 boundary edges
 *         (non-manifold boundary). Interior valences are not validated
 *         (any @f$n \geq 3@f$ is accepted; @f$n = 2@f$ would arise only
 *         on a pinched non-manifold mesh).
 */
[[nodiscard]] LoopSubdivision loop_subdivide_one_step(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

/**
 * @brief Apply Loop subdivision @p n_passes times in succession.
 *
 * Composes @ref loop_subdivide_one_step @p n_passes times. Each
 * additional pass shrinks the residual irregular-area approximation in
 * the L.3.4 path by a factor of 1/4 (1 pass: 25% dropped, 2 passes:
 * 6.25%, 3 passes: 1.5625%).
 *
 * The composed constraint matrix is
 * @f$ S = S_{n-1} S_{n-2} \cdots S_0 @f$, where each @f$ S_i @f$ is the
 * matrix returned by the @f$ i @f$-th single subdivision. @c n_real and
 * @c n_real_faces refer to the original (input) vertex / face counts;
 * @c V_sub.rows() and @c F_sub.rows() refer to the final subdivided
 * mesh's totals.
 *
 * @param V         Input vertex matrix, @f$ n_\text{real} \times 3 @f$.
 * @param F         Input face matrix, @f$ n_\text{real\_faces} \times 3 @f$.
 * @param n_passes  Number of subdivision passes; must be @f$ \ge 0 @f$.
 *                  Zero passes returns the input verbatim with @c S
 *                  set to the identity.
 *
 * @return Populated @ref LoopSubdivision after @p n_passes subdivisions.
 *
 * @throws std::invalid_argument on negative @p n_passes.
 * @throws std::runtime_error on the same conditions as
 *         @ref loop_subdivide_one_step.
 */
[[nodiscard]] LoopSubdivision loop_subdivide_n_times(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int                    n_passes);

// ---------------------------------------------------------------------------
// Global stiffness assembly on the augmented mesh.
// ---------------------------------------------------------------------------

/**
 * @brief Assemble the global stiffness matrix on the augmented mesh.
 *
 * Walks every real triangle of @c aug.F_aug (rows
 * @c 0 .. n_real_faces-1), builds the patch stencil on the augmented
 * mesh, verifies all 3 corners have augmented valence 6, computes the
 * canonical Fig. 9 12-vertex DOF list with @ref canonical_regular_dofs,
 * evaluates the limit surface at the element centroid
 * (@f$v = w = 1/3@f$, one-point quadrature), builds the @f$36 \times 36@f$
 * element stiffness with @ref element_stiffness_regular, and scatters
 * into a sparse @f$3 n_\text{aug} \times 3 n_\text{aug}@f$ matrix.
 *
 * Phantom triangles do not contribute material energy — they exist
 * solely so that the augmented directed-edge map is 1-to-1 (necessary
 * for the CCW topology walk to find every spoke).
 *
 * @param aug                       Output of @ref augment_for_loop_boundary.
 * @param material                  @ref ShellMaterial with @c k_L, @c k_B,
 *                                  @c poisson_ratio.
 * @param skip_irregular_triangles  If false (default), the function throws
 *                                  on any non-valence-6 augmented corner.
 *                                  If true, such triangles are silently
 *                                  skipped and contribute zero energy.
 *                                  The skip path is used by L.3.4's
 *                                  one-step subdivision pipeline: after
 *                                  one global Loop subdivision, only the
 *                                  residual sub-triangles adjacent to the
 *                                  original extraordinary vertices remain
 *                                  irregular (with area shrunk to 1/4 of
 *                                  parent), and the Cirak-Ortiz Section
 *                                  4.6 step 1 approximation simply drops
 *                                  their contribution. Ignored when
 *                                  @p use_stam_for_irregular is true.
 * @param use_stam_for_irregular    If true, irregular triangles (exactly
 *                                  one corner with valence @f$ \ne 6 @f$,
 *                                  others valence 6) are assembled via
 *                                  the S.6 Stam exact-evaluation kernel
 *                                  @ref element_stiffness_stam instead
 *                                  of being thrown / skipped. The
 *                                  @c StamEvaluator is built once per
 *                                  observed valence and cached for the
 *                                  duration of this call. Triangles with
 *                                  more than one extraordinary corner
 *                                  still trigger the throw / skip path
 *                                  (these should not exist after one
 *                                  Loop subdivision of an isolated-
 *                                  extraordinary-vertex mesh).
 *
 * @param rule  Reference-triangle quadrature rule for the per-element
 *              @f$K@f$ integrals.
 *
 * @return Sparse symmetric @f$K_\text{aug}@f$ on the augmented DOF layout.
 *
 * @throws std::runtime_error
 *         if @p skip_irregular_triangles is false, @p use_stam_for_irregular
 *         is false, and any real triangle has a non-valence-6 corner in
 *         the augmented mesh, or on degenerate parametrisation; or if
 *         @p use_stam_for_irregular is true and any irregular triangle
 *         has more than one extraordinary corner (i.e. the input mesh
 *         has adjacent extraordinary vertices that have not been
 *         isolated by Loop subdivision yet).
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_stiffness_augmented(
    const LoopAugmented& aug,
    const ShellMaterial& material,
    bool                 skip_irregular_triangles = false,
    bool                 use_stam_for_irregular   = false,
    QuadratureRule       rule                     = QuadratureRule::SevenPointDunavant);

/**
 * @brief Loop-subdivision shell stiffness matrix at rest.
 *
 * Top-level entry point. Two code paths:
 *  - **Fast path** — input has no extraordinary interior vertices
 *    (every interior vertex has valence 6). Runs:
 *    @ref augment_for_loop_boundary @f$\to@f$
 *    @ref assemble_stiffness_augmented @f$\to@f$ @f$K = C^\top K_\text{aug} C@f$.
 *  - **Subdivision path** — input has at least one extraordinary
 *    interior vertex. Runs
 *    @ref loop_subdivide_n_times first (n_passes Loop steps) to push
 *    the irregular area down to a small residual per parent irregular
 *    triangle (1/4^n of original area), then
 *    @ref augment_for_loop_boundary @f$\to@f$
 *    @ref assemble_stiffness_augmented @f$\to@f$
 *    @f$K = S^\top C^\top K_\text{sub,aug} C\,S@f$. The residual
 *    irregular sub-triangles are handled by one of two policies:
 *    + Default (L.3.4 step-1 multi-pass) — drop their contribution.
 *      Geometric convergence with refinement, ratio ≈ 0.55 per step.
 *    + S-series Stam exact-eval (@p use_stam = true) — evaluate via
 *      the Stam 1999 eigenbasis. Theoretically exact on the irregular
 *      sub-tile; empirically ≈ L.3.4 on sphere meshes.
 *
 * Both paths return a stiffness matrix on the input DOF layout
 * (@f$3 n_\text{real}@f$ entries) — the subdivision path's @c S and
 * @c C reductions are transparent to the caller.
 *
 * @param V         Vertex matrix @f$n_\text{real} \times 3@f$.
 * @param F         Face matrix @f$n_\text{real\_faces} \times 3@f$.
 * @param material  @ref ShellMaterial calibration.
 * @param n_passes  Number of subdivision passes used in the irregular
 *                  branch (default 1). Ignored when no extraordinary
 *                  interior vertices are present (the fast path skips
 *                  subdivision entirely). Each extra pass shrinks the
 *                  residual irregular-area approximation by a factor
 *                  of 1/4. Must be @f$ \ge 1 @f$. Under the Stam path
 *                  (@p use_stam = true) only the FIRST subdivision pass
 *                  is needed to isolate extraordinary vertices; extra
 *                  passes still refine the patch but contribute no
 *                  additional accuracy from the kernel itself (Stam
 *                  evaluation is exact at any depth).
 * @param use_stam  If true and the mesh has extraordinary interior
 *                  vertices, use the S-series Stam exact-evaluation
 *                  kernel for irregular sub-triangles instead of the
 *                  L.3.4 step-1 multi-pass drop approximation. The
 *                  pipeline becomes
 *                  @ref loop_subdivide_n_times @f$\to@f$
 *                  @ref augment_for_loop_boundary @f$\to@f$
 *                  @ref assemble_stiffness_augmented (with
 *                  @c use_stam_for_irregular = true) @f$\to@f$
 *                  @f$K = S^\top C^\top K_\text{sub,aug} C\,S@f$.
 *                  Default false — the S.8 fixture comparison found
 *                  Stam and the L.3.4 approximation equivalent in
 *                  practice, so L.3.4 stays the default and Stam is
 *                  kept opt-in.
 *
 * @param rule  Reference-triangle quadrature rule for the per-element
 *              @f$K@f$ integrals (forwarded to
 *              @ref assemble_stiffness_augmented).
 *
 * @return Sparse symmetric @f$K \in \mathbb R^{3 n_\text{real} \times 3 n_\text{real}}@f$
 *         in the same DOF layout as the existing
 *         @ref chladni::shell::assemble_stiffness_at_rest_analytic.
 *         Rigid translations annihilate @c K exactly (built into the
 *         element formulation and preserved by @c C and @c S).
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_stiffness_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ShellMaterial&   material,
    int                    n_passes = 1,
    bool                   use_stam = false,
    QuadratureRule         rule     = QuadratureRule::SevenPointDunavant);

/**
 * @brief Augmented-mesh consistent mass matrix on the Loop limit surface.
 *
 * Mirrors @ref assemble_stiffness_augmented exactly: same patch-stencil
 * construction, same regular/irregular dispatch, same Stam-or-drop
 * policy for residual irregular sub-triangles. Per-element matrices
 * come from @ref element_mass_regular / @ref element_mass_stam and are
 * scattered into a sparse @f$3 n_\text{aug} \times 3 n_\text{aug}@f$
 * matrix on the augmented DOF layout.
 *
 * **Why mirror K's dispatch?** The generalized eigenproblem
 * @f$K v = \omega^2 M v@f$ only makes sense if @c K and @c M are
 * assembled on the *same* discrete function space. If @c K uses Stam
 * exact evaluation on an irregular sub-triangle while @c M drops it
 * (or vice versa), the test functions for the two matrices differ and
 * spurious modes appear.
 *
 * @param aug                       Augmented mesh from @ref augment_for_loop_boundary.
 * @param surface_density           @f$\rho h@f$ (kg/m²), @f$> 0@f$.
 * @param skip_irregular_triangles  Drop element contributions from real
 *                                  faces with one or more irregular
 *                                  corners. Must match the K-side flag.
 * @param use_stam_for_irregular    Use Stam exact evaluation for
 *                                  irregular faces (requires exactly
 *                                  one extraordinary corner per face).
 *                                  Must match the K-side flag.
 *
 * @param rule  Reference-triangle quadrature rule for the per-element
 *              @f$M@f$ integrals.
 *
 * @return Sparse symmetric @f$M_\text{aug} \in \mathbb R^{3 n_\text{aug} \times 3 n_\text{aug}}@f$.
 *
 * @throws std::runtime_error on an irregular face when neither
 *         @p skip_irregular_triangles nor @p use_stam_for_irregular is
 *         true; @ref element_mass_regular / @ref element_mass_stam may
 *         also throw on degenerate parametrisations.
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_mass_augmented(
    const LoopAugmented& aug,
    double               surface_density,
    bool                 skip_irregular_triangles = false,
    bool                 use_stam_for_irregular   = false,
    QuadratureRule       rule                     = QuadratureRule::SevenPointDunavant);

/**
 * @brief Loop-subdivision shell consistent mass matrix.
 *
 * Top-level entry point. Mirrors @ref assemble_stiffness_loop exactly:
 *  - **Fast path** (no extraordinary interior vertices):
 *    @ref augment_for_loop_boundary @f$\to@f$
 *    @ref assemble_mass_augmented @f$\to@f$ @f$M = C^\top M_\text{aug} C@f$.
 *  - **Subdivision path** (any extraordinary interior vertex):
 *    @ref loop_subdivide_n_times (n_passes) @f$\to@f$
 *    @ref augment_for_loop_boundary @f$\to@f$
 *    @ref assemble_mass_augmented (with @c skip_irregular_triangles or
 *    @c use_stam_for_irregular per @p use_stam) @f$\to@f$
 *    @f$M = S^\top C^\top M_\text{sub,aug} C\,S@f$.
 *
 * Both paths return @f$M@f$ in the same DOF layout as
 * @ref assemble_stiffness_loop — the two are designed to be used as a
 * pair in the generalized eigenproblem
 * @f$K v = \omega^2 M v@f$.
 *
 * @param V               Vertex matrix @f$n_\text{real} \times 3@f$.
 * @param F               Face matrix @f$n_\text{real\_faces} \times 3@f$.
 * @param surface_density @f$\rho h@f$ (kg/m²), @f$> 0@f$.
 * @param n_passes        Subdivision passes in the irregular branch.
 *                        Must match the K-side value.
 * @param use_stam        Stam exact eval for residual irregular
 *                        sub-triangles. Must match the K-side flag.
 * @param rule            Reference-triangle quadrature rule for the
 *                        per-element @f$M@f$ integrals. Should match
 *                        the K-side choice.
 *
 * @return Sparse symmetric positive-semi-definite
 *         @f$M \in \mathbb R^{3 n_\text{real} \times 3 n_\text{real}}@f$.
 *
 * @throws std::invalid_argument on @p n_passes @f$< 1@f$ or
 *         @p surface_density @f$\le 0@f$.
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_mass_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double                 surface_density,
    int                    n_passes = 1,
    bool                   use_stam = false,
    QuadratureRule         rule     = QuadratureRule::SevenPointDunavant);

// ---------------------------------------------------------------------------
// S-series: Stam 1999 exact eigenstructure of the irregular subdivision matrix.
//
// Replacement for the L.3.4 step-1 multi-pass approximation. The full
// S-series (S.1-S.7) is shipped and wired: offline eigendecomposition,
// tile mapping, eigenbasis evaluation, derivatives, and per-element K
// assembly (use_stam_for_irregular in the element-K path). Opt-in via
// LoopAssembler::Params::use_stam.
// ---------------------------------------------------------------------------

/**
 * @brief Eigendecomposition of the extended subdivision matrix
 *        @f$ A \in \mathbb R^{K \times K} @f$ around an extraordinary
 *        vertex of valence @f$ N @f$, with @f$ K = N + 6 @f$.
 *
 * Stam 1999 Section 3.2 + Appendix B. The matrix
 * @f[
 *    A \;=\; \begin{pmatrix} S & 0 \\ S_{11} & S_{12} \end{pmatrix},
 * @f]
 * with @f$ S @f$ of size @f$ (N+1) \times (N+1) @f$ (the extraordinary
 * vertex + its @f$ N @f$-vertex 1-ring), @f$ S_{11} @f$ of size
 * @f$ 5 \times (N+1) @f$, and @f$ S_{12} @f$ of size @f$ 5 \times 5 @f$
 * (independent of @f$ N @f$).
 *
 * Because @f$ A @f$ is block-upper-triangular,
 * @f$ \sigma(A) \;=\; \sigma(S) \cup \sigma(S_{12}) @f$ as a multiset.
 * For @f$ N \ge 4 @f$ the eigenvectors of @f$ A @f$ are built from those
 * of @f$ S @f$ (with the bottom @f$ S_{12} @f$-related 5 components
 * determined by Stam eq. (4)) plus the eigenvectors of @f$ S_{12} @f$
 * (embedded into the bottom 5 rows of @f$ A @f$ with zero top).
 *
 * The valence @f$ N = 3 @f$ case is special: @f$ \mu_2 = \alpha(3)
 * \;=\; 1/16 @f$ coincides with an eigenvalue of @f$ S_{12} @f$, so the
 * linear system that determines the corresponding column of
 * @f$ \mathbf U_1 @f$ becomes degenerate. @f$ A @f$ has a non-trivial
 * Jordan block of size 2 at @f$ \lambda = 1/16 @f$ (Stam Appendix C);
 * the matrices below store the Jordan basis explicitly, with the
 * super-diagonal "1" implicit (signalled by @c has_jordan_block).
 *
 * @section storage Storage
 *
 *  - @c lambda: length @f$ K @f$ vector of diagonal entries of
 *    @f$ \Lambda @f$ (or @f$ J @f$ when @c has_jordan_block is true).
 *  - @c V: @f$ K \times K @f$ matrix whose columns are the
 *    (generalized) eigenvectors, in the same order as @c lambda.
 *  - @c V_inv: numerical inverse of @c V, precomputed for use during
 *    parameter-space evaluation (Stam's `ProjectPoints` routine).
 *
 * For @f$ N \ne 3 @f$, @f$ A = V \cdot \mathrm{diag}(\lambda) \cdot
 * V_{\text{inv}} @f$ holds exactly up to numerical roundoff.
 * For @f$ N = 3 @f$, @f$ A = V \cdot J \cdot V_{\text{inv}} @f$ where
 * @f$ J = \mathrm{diag}(\lambda) + e_{K-2,K-1} @f$ (a 1 on the
 * super-diagonal in the last column).
 *
 * @section stam_refs References
 *  - @cite stam_1999_loop_evaluation Section 3.2 (eigenstructure of A),
 *    Appendix B (explicit @f$ S @f$, @f$ S_{11} @f$, @f$ S_{12} @f$
 *    blocks and the Fourier eigenanalysis of @f$ S @f$), Appendix C
 *    (@f$ N = 3 @f$ Jordan-block case).
 */
struct StamEigenstructure {
    int N;                       ///< Extraordinary-vertex valence, @f$ \ge 3 @f$.
    bool has_jordan_block;       ///< True iff @c N == 3 (App. C Jordan block at 1/16).
    Eigen::VectorXd lambda;      ///< Length @f$ K = N + 6 @f$.
    Eigen::MatrixXd V;           ///< @f$ K \times K @f$ (generalized) eigenvectors.
    Eigen::MatrixXd V_inv;       ///< @f$ K \times K @f$ numerical inverse of @c V.
};

/**
 * @brief Explicit @f$ K \times K @f$ extended subdivision matrix
 *        @f$ A @f$ around an extraordinary vertex of valence @f$ N @f$.
 *
 * Useful for cross-validation against the eigendecomposition. Built
 * directly from the Loop masks (Stam Appendix B) — does not depend
 * on @ref stam_eigenstructure.
 *
 * Index layout (columns and rows both):
 *  - @c 0: the extraordinary (central) vertex.
 *  - @c 1..N: the @f$ N @f$ 1-ring vertices around the extraordinary
 *    vertex, in cyclic order.
 *  - @c N+1..N+5: the five outer vertices completing the
 *    @f$ K = N + 6 @f$ control points of the regular triangular
 *    box-spline patch shown in Stam Fig. 2.
 *
 * @throws std::invalid_argument on @c N < 3.
 */
[[nodiscard]] Eigen::MatrixXd
build_extended_subdivision_matrix(int N);

/**
 * @brief Stam's @b bigger extended subdivision matrix @f$ \bar A @f$
 *        of size @f$ M \times K = (N+12) \times (N+6) @f$.
 *
 * Stam 1999 Eq. (1) defines two subdivision matrices around an
 * extraordinary vertex of valence @f$ N @f$:
 *  - @f$ A @f$ of size @f$ K \times K @f$ (the "extraordinary rules"),
 *    produced by @ref build_extended_subdivision_matrix.
 *  - @f$ \bar A @f$ of size @f$ M \times K @f$, which adds 6 more
 *    rows so that the @f$ M = N + 12 @f$ vertices output by
 *    @f$ \bar C_1 = \bar A C_0 @f$ contain enough information to
 *    evaluate the three regular sub-patches (Stam Fig. 3, Fig. 4).
 *
 * Block decomposition (Stam Eq. (1)):
 * @f[
 *    \bar A \;=\; \begin{pmatrix} S & 0 \\ S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix},
 * @f]
 * with @f$ S_{21} @f$ of size @f$ 6 \times (N+1) @f$ and @f$ S_{22} @f$
 * of size @f$ 6 \times 5 @f$, both with closed-form entries given in
 * Stam App. B (independent of @f$ N @f$ in @f$ S_{22} @f$'s case;
 * @f$ S_{21} @f$ has only six nonzero columns at positions 1, 2,
 * @f$ N-1 @f$, and @f$ N @f$).
 *
 * Index layout:
 *  - Rows @c 0..N: the top block @f$ S @f$.
 *  - Rows @c N+1..N+5: the middle block @f$ [S_{11}\ S_{12}] @f$.
 *  - Rows @c N+6..N+11: the new bottom block @f$ [S_{21}\ S_{22}] @f$.
 *  - Columns @c 0..N: extraordinary vertex + N 1-ring vertices.
 *  - Columns @c N+1..N+5: five outer vertices.
 *
 * @throws std::invalid_argument on @c N < 3.
 */
[[nodiscard]] Eigen::MatrixXd
build_extended_subdivision_matrix_bar(int N);

/**
 * @brief Compute @ref StamEigenstructure for valence @f$ N \ge 3 @f$.
 *
 * Strategy (per the project memory's "analytical for S, numerical for
 * blocks" decision):
 *  - Eigendecompose @f$ S @f$ analytically via Fourier diagonalisation
 *    of the cyclic 1-ring (Stam App. B). The eigenvalues are
 *    @f$ \mu_1 = 1 @f$, @f$ \mu_2 = 5/8 - \alpha(N) @f$, and
 *    @f$ \mu_{3+k} = f(k) = 3/8 + (1/4)\cos(2\pi k / N) @f$ for
 *    @f$ k = 1, \ldots, N - 1 @f$ (with @f$ \alpha(N) = 5/8 - (3 +
 *    2\cos(2\pi/N))^2 / 64 @f$).
 *  - Eigendecompose @f$ S_{12} @f$ numerically via @c Eigen::EigenSolver
 *    (5x5, constant in @f$ N @f$; computed once and used as-is).
 *  - For each @f$ S @f$-eigenvalue @f$ \sigma @f$ with eigenvector
 *    @f$ u_0 @f$, solve @f$ (\sigma I - S_{12}) u_1 = S_{11} u_0 @f$
 *    for the corresponding bottom 5 components.
 *  - When @f$ \sigma @f$ collides with a @f$ S_{12} @f$-eigenvalue (the
 *    @f$ N @f$-even @f$ \sigma = 1/8 @f$ case from Stam App. B), use
 *    the explicit closed-form bottom column @f$ u_{1,N+1} =
 *    (0, 8, 0, -8, 0)^\top @f$.
 *  - For @f$ N = 3 @f$, return the explicit Jordan basis @f$ V @f$ and
 *    its inverse from Stam Appendix C.
 *
 * @throws std::invalid_argument on @c N < 3.
 *
 * @return @ref StamEigenstructure satisfying @f$ V \Lambda V^{-1} = A @f$
 *         (for @f$ N \ne 3 @f$) or @f$ V J V^{-1} = A @f$ (for
 *         @f$ N = 3 @f$, with the implicit Jordan super-diagonal).
 */
[[nodiscard]] StamEigenstructure
stam_eigenstructure(int N);

/**
 * @brief Result of @ref stam_tile_map — which dyadic sub-tile of the
 *        unit triangle a parameter @f$ (v, w) @f$ lies in, plus the
 *        affine map of @f$ (v, w) @f$ onto the unit triangle.
 */
struct StamTileMap {
    int n;       ///< Subdivision level, @f$ \ge 1 @f$.
    int k;       ///< Tile index, @f$ k \in \{1, 2, 3\} @f$ (Stam Sec 3).
    double v_p;  ///< Mapped parameter, @f$ v' \in [0, 1] @f$.
    double w_p;  ///< Mapped parameter, @f$ w' \in [0, 1 - v'] @f$.
};

/**
 * @brief Map a parameter @f$ (v, w) \in \Omega @f$ onto one of the
 *        dyadic sub-tiles @f$ \Omega_k^n @f$ of Stam Sec 3 and return
 *        the corresponding affine reparametrisation onto the unit
 *        triangle.
 *
 * The unit triangle @f$ \Omega = \{(v, w) \mid v \in [0, 1],
 * w \in [0, 1 - v]\} @f$ is partitioned (Stam Fig. 5) into an infinite
 * set of triangular tiles @f$ \Omega_k^n @f$ for @f$ n \ge 1 @f$ and
 * @f$ k \in \{1, 2, 3\} @f$, defined by
 * @f[
 *   \begin{aligned}
 *     \Omega_1^n &= \{ (v, w) \mid v \in [2^{-n}, 2^{-n+1}],
 *                                  w \in [0, 2^{-n+1} - v] \}, \\
 *     \Omega_2^n &= \{ (v, w) \mid v \in [0, 2^{-n}],
 *                                  w \in [0, 2^{-n} - v] \}_{\text{flipped}}, \\
 *     \Omega_3^n &= \{ (v, w) \mid v \in [0, 2^{-n}],
 *                                  w \in [2^{-n}, 2^{-n+1} - v] \}.
 *   \end{aligned}
 * @f]
 * The tiles cover the open triangle (the point @f$ (0, 0) @f$ — the
 * extraordinary vertex — is the limit of @f$ n \to \infty @f$ and is
 * not in any tile). Each tile is affinely mapped onto the unit
 * triangle by
 * @f[
 *   \begin{aligned}
 *     t_{n,1}(v, w) &= (2^n v - 1, 2^n w), \\
 *     t_{n,2}(v, w) &= (1 - 2^n v, 1 - 2^n w), \\
 *     t_{n,3}(v, w) &= (2^n v, 2^n w - 1).
 *   \end{aligned}
 * @f]
 *
 * @section evalsurf_algorithm Algorithm
 *
 * Following Stam Sec 4 @c EvalSurf:
 *  1. Subdivision level @f$ n = \lfloor 1 - \log_2(v + w) \rfloor @f$
 *     (n=1 if @f$ v + w \in (1/2, 1] @f$, n=2 if @f$ \in (1/4, 1/2] @f$, etc).
 *  2. Rescale: @f$ v' = 2^{n-1} v @f$, @f$ w' = 2^{n-1} w @f$.
 *  3. Branch:
 *     - if @f$ v' > 1/2 @f$: @c k = 1, @f$ v_p = 2 v' - 1 @f$, @f$ w_p = 2 w' @f$.
 *     - else if @f$ w' > 1/2 @f$: @c k = 3, @f$ v_p = 2 v' @f$, @f$ w_p = 2 w' - 1 @f$.
 *     - else: @c k = 2, @f$ v_p = 1 - 2 v' @f$, @f$ w_p = 1 - 2 w' @f$.
 *
 * The branch order matters at tile boundaries (we follow Stam's
 * preference: @c k=1 strict-takes-priority over @c k=3, both over
 * @c k=2).
 *
 * @throws std::invalid_argument
 *   on @f$ v < 0 @f$, @f$ w < 0 @f$, @f$ v + w > 1 + \epsilon @f$
 *   (with @f$ \epsilon @f$ a small floating-point tolerance), or
 *   @f$ v + w @f$ at the machine-zero limit (the extraordinary-vertex
 *   degenerate case where no finite tile contains @f$ (v, w) @f$).
 *
 * Reference: @cite stam_1999_loop_evaluation Sec 3 (tile partition,
 * Fig. 5) and Sec 4 (`EvalSurf` algorithm).
 */
[[nodiscard]] StamTileMap stam_tile_map(double v, double w);

/**
 * @brief The three picking matrices @f$ P_1, P_2, P_3 @f$ that select the
 *        twelve regular box-spline control points of each regular dyadic
 *        sub-patch out of the @f$ M = N + 12 @f$ control points produced
 *        by @ref build_extended_subdivision_matrix_bar.
 *
 * Stam Eq. (2) writes the surface restricted to the dyadic tile
 * @f$ \Omega_k^n @f$ as
 * @f[
 *   s\big|_{\Omega_k^n}(v, w)
 *   = C_0^\top (P_k\, \bar A\, A^{n-1})^\top b\big(t_{n,k}(v, w)\big),
 * @f]
 * with @f$ b @f$ the 12-vector of regular box-spline basis values
 * (@ref regular_basis) on the unit triangle, @f$ A @f$ the extended
 * subdivision matrix (@ref build_extended_subdivision_matrix), and
 * @f$ \bar A @f$ its 6-row extension (@ref build_extended_subdivision_matrix_bar).
 * Each @f$ P_k @f$ is a @f$ 12 \times M @f$ 0/1 selection matrix with
 * exactly one @c 1 per row.
 */
struct StamPickingMatrices {
    int N;                                                          ///< Valence, @f$ \ge 3 @f$.
    std::array<Eigen::Matrix<double, 12, Eigen::Dynamic>, 3> P;     ///< @c P[k-1] is @f$ P_k @f$, of shape @f$ 12 \times (N+12) @f$.
};

/**
 * @brief Build the three picking matrices @f$ P_1, P_2, P_3 @f$ for an
 *        extraordinary patch of valence @f$ N \ge 3 @f$.
 *
 * @section semantics Tile semantics (Stam Sec 3, Fig. 5)
 *
 * The central irregular triangle of the patch — corners @f$(0, 1, N)@f$
 * in the @f$ K = N + 6 @f$ control-vertex labelling — is parametrised by
 * @f$ (v, w) \in \Omega @f$ with the extraordinary vertex @c 0 at
 * @f$ (v, w) = (0, 0) @f$, vertex @c 1 at @f$ (1, 0) @f$, and vertex
 * @c N at @f$ (0, 1) @f$. Under one Loop subdivision step the central
 * triangle splits into four sub-triangles; three of them are regular
 * (the corner-1 sub-tile, the corner-N sub-tile, and the medial
 * sub-tile) and one is irregular (the sub-tile touching the
 * extraordinary vertex).
 *
 * The picking matrices index those three regular sub-tiles by Stam's
 * tile label @f$ k \in \{1, 2, 3\} @f$:
 *  - @c P[0] @f$\equiv P_1@f$: corner sub-tile near @f$ v = 1 @f$
 *    (i.e. corner at the @f$ k = 1 @f$-th 1-ring vertex).
 *  - @c P[1] @f$\equiv P_2@f$: medial sub-tile.
 *  - @c P[2] @f$\equiv P_3@f$: corner sub-tile near @f$ w = 1 @f$
 *    (corner at the @f$ N @f$-th 1-ring vertex).
 *
 * The 12 selected control points per tile are arranged in canonical
 * Cirak-Ortiz Fig. 9 slot order (corners at slots 3, 6, 7), matching
 * @ref regular_basis and @ref canonical_regular_dofs.
 *
 * @section picking_algorithm Algorithm (topology-driven)
 *
 * Following the @c project_chladni_stam_refactor_plan handoff:
 *  1. Construct the irregular patch's @f$ K = N + 6 @f$ control vertices
 *     and its @f$ N + 7 @f$ faces (closed @f$ N @f$-disk around the
 *     extraordinary vertex plus 7 outer faces completing the box-spline
 *     neighbourhood of the central triangle, mirroring Stam Fig. 2).
 *     Vertex labelling matches @ref build_extended_subdivision_matrix_bar —
 *     @c 0 = extraordinary, @c 1..N = 1-ring, @c N+1..N+5 = outer 5.
 *  2. Apply one Loop subdivision step (@ref loop_subdivide_one_step).
 *  3. Identify the three regular sub-faces of the central triangle in
 *     the subdivided face list.
 *  4. For each regular sub-face, get the 12 canonical-Fig.-9-slot
 *     control points via @ref canonical_regular_dofs.
 *  5. Map each subdivided-mesh vertex index to its row index in
 *     @f$ \bar A @f$ via the @f$ (v_\text{sub} \to \text{Stam-}M) @f$
 *     correspondence implied by @ref build_extended_subdivision_matrix_bar's
 *     row blocks (top @f$ N + 1 @f$ rows = smoothed centre + @f$ N @f$
 *     central-spoke midpoints; middle 5 rows = new outer ring around
 *     the central triangle; bottom 6 rows = outer-extension midpoints
 *     listed in @c S_21 / @c S_22).
 *  6. Each canonical DOF lands in exactly one of the @f$ M = N + 12 @f$
 *     Stam indices; populate @f$ P_k(\text{slot}, \text{Stam-idx}) = 1 @f$.
 *
 * @throws std::invalid_argument on @c N < 3.
 * @throws std::runtime_error if any canonical DOF falls outside Stam's
 *         @f$ M @f$-set (programmer error: indicates a topology /
 *         indexing mismatch between this routine and
 *         @ref build_extended_subdivision_matrix_bar).
 *
 * @return @ref StamPickingMatrices with @c P[0..2] populated.
 */
[[nodiscard]] StamPickingMatrices
stam_picking_matrices(int N);

/**
 * @brief Cached state for fast Stam-eigenbasis @f$ \Phi(v, w) @f$
 *        evaluation around an extraordinary vertex of valence
 *        @f$ N \ge 3 @f$.
 *
 * Holds the eigenvalues @f$ \lambda @f$ and the three pre-multiplied
 * matrices @f$ (P_k\, \bar A\, V)^\top @f$ (one per tile @f$ k @f$),
 * each of shape @f$ K \times 12 @f$ with @f$ K = N + 6 @f$. All three
 * depend only on @f$ N @f$, so the heavy work happens once per valence
 * and per-evaluation cost reduces to a 12-d dot product per @f$ \Phi @f$
 * component plus an @f$ \mathcal O(K) @f$ scaling pass.
 */
struct StamEvaluator {
    int N;                                                              ///< Valence, @f$ \ge 3 @f$.
    bool has_jordan_block;                                              ///< True iff @c N == 3 (Stam App. C Jordan basis).
    Eigen::VectorXd lambda;                                             ///< Length @f$ K = N + 6 @f$.
    std::array<Eigen::Matrix<double, Eigen::Dynamic, 12>, 3> Pk_Abar_V_T;
    ///< @c Pk_Abar_V_T[k-1] is @f$ (P_k\, \bar A\, V)^\top @f$ for tile @f$ k \in \{1, 2, 3\} @f$, of shape @f$ K \times 12 @f$.
    Eigen::MatrixXd V_inv_T;                                            ///< @f$ V^{-\top} @f$, shape @f$ K \times K @f$. Used by @ref evaluate_patch_stam to convert eigenbasis @f$ \Phi @f$ to node basis @f$ \eta = V^{-\top} \Phi @f$.
};

/**
 * @brief Build a @ref StamEvaluator for valence @f$ N \ge 3 @f$.
 *
 * Composes @ref stam_eigenstructure, @ref build_extended_subdivision_matrix_bar
 * and @ref stam_picking_matrices, then materialises the three
 * @f$ (P_k\, \bar A\, V)^\top @f$ matrices.
 *
 * @throws std::invalid_argument on @c N < 3.
 */
[[nodiscard]] StamEvaluator make_stam_evaluator(int N);

/**
 * @brief Evaluate the Stam eigenbasis @f$ \Phi(v, w) @f$ at a parameter
 *        in the central irregular triangle's unit triangle.
 *
 * Implements Stam Eq. (5):
 * @f[
 *   \Phi(v, w)\big|_{\Omega_k^n}
 *   \;=\; \Lambda^{n-1}\, (P_k\, \bar A\, V)^\top\, b\big(t_{n,k}(v, w)\big),
 * @f]
 * with the dyadic tile lookup @f$ (n, k, v_p, w_p) = @f$
 * @ref stam_tile_map @c (v, w) and the regular box-spline basis
 * @f$ b @f$ = @ref regular_basis. Surface evaluation from K control
 * scalars @f$ C_0 @f$ is then @f$ s(v, w) = \hat C_0^\top \Phi(v, w) @f$
 * with @f$ \hat C_0 = V^{-1} C_0 @f$ (Stam Eq. (6)).
 *
 * @section jordan N = 3 Jordan block
 *
 * For @c N == 3 the @f$ \lambda = 1/16 @f$ block has algebraic
 * multiplicity 3 / geometric multiplicity 2 (Stam App. C). Tracing
 * Stam Eq. (5) back to Eq. (2) shows that "@f$ \Lambda^{n-1} @f$"
 * really means @f$ (J^\top)^{n-1} @f$, so the @f$ (n - 1)\,
 * \lambda^{n-2} @f$ off-diagonal contribution lives at the
 * sub-diagonal entry @f$ (K-1, K-2) @f$ — lifting @f$ \Phi(K-1) @f$
 * by @f$ (n-1) \lambda^{n-2} @f$ times the pre-scaling @f$ u(K-2) @f$
 * row. This branch is wired into the impl via
 * @c apply_lambda_pow_and_jordan_in_place; @c N == 3 callers see the
 * exact limit surface across all (n, k).
 *
 * @throws std::invalid_argument
 *   on @f$ v < 0 @f$, @f$ w < 0 @f$, @f$ v + w > 1 + \epsilon @f$, or
 *   @f$ v + w @f$ at the machine-zero limit (the extraordinary-vertex
 *   degenerate case where no finite tile contains @f$ (v, w) @f$).
 *
 * @return Length-@c K = N + 6 vector @f$ \Phi(v, w) @f$.
 */
[[nodiscard]] Eigen::VectorXd stam_phi(
    const StamEvaluator& ev, double v, double w);

/**
 * @brief First partials @f$ \partial \Phi / \partial v @f$ and
 *        @f$ \partial \Phi / \partial w @f$ of the Stam eigenbasis at
 *        a parameter @f$ (v, w) @f$ in the central irregular triangle.
 *
 * @section chain_rule Chain rule through the affine tile transform
 *
 * Differentiating Stam Eq. (5) wrt the *global* (v, w) and using the
 * affine tile transform @f$ t_{n,k} @f$ (Stam Sec 3 / Fig. 5):
 * @f[
 *   \frac{\partial \Phi}{\partial v}\bigg|_{\Omega_k^n}
 *   = \Lambda^{n-1}\, (P_k\, \bar A\, V)^\top
 *     \frac{\partial b}{\partial v_p}(t_{n,k}(v, w))
 *     \cdot \frac{\partial v_p}{\partial v},
 * @f]
 * with @f$ \frac{\partial v_p}{\partial v} = \sigma_k \cdot 2^n @f$
 * (and analogously for @f$ w_p, w @f$). The per-tile sign is
 * @f$ \sigma_1 = +1 @f$, @f$ \sigma_2 = -1 @f$, @f$ \sigma_3 = +1 @f$
 * (the medial tile @f$ \Omega_2 @f$ is "flipped"; the corner tiles
 * @f$ \Omega_1, \Omega_3 @f$ are not).
 *
 * Because the tile Jacobian is diagonal (@f$ \partial v_p / \partial w
 * = \partial w_p / \partial v = 0 @f$), the @f$ v @f$- and
 * @f$ w @f$-partials decouple — each picks up exactly one
 * @c regular_basis_grad column scaled by @f$ \sigma_k \cdot 2^n @f$.
 *
 * @section ret Return shape
 *
 * @f$ K \times 2 @f$, with column 0 = @f$ \partial \Phi / \partial v @f$
 * and column 1 = @f$ \partial \Phi / \partial w @f$, both length
 * @f$ K = N + 6 @f$.
 *
 * @section jordan_grad N = 3 Jordan block
 *
 * Same as @ref stam_phi — the Jordan correction is applied by the
 * shared @c apply_lambda_pow_and_jordan_in_place helper, so the
 * gradient is exact for @c N == 3 across all (n, k) too.
 *
 * @throws std::invalid_argument on the same conditions as
 *         @ref stam_tile_map.
 */
[[nodiscard]] Eigen::Matrix<double, Eigen::Dynamic, 2>
stam_phi_grad(const StamEvaluator& ev, double v, double w);

/**
 * @brief Second partials @f$ \partial^2 \Phi / \partial v^2 @f$,
 *        @f$ \partial^2 \Phi / \partial v \partial w @f$,
 *        @f$ \partial^2 \Phi / \partial w^2 @f$ of the Stam eigenbasis.
 *
 * @section chain_rule_hess Chain rule
 *
 * Differentiating Stam Eq. (5) twice and using the affine tile
 * transform @f$ t_{n,k} @f$:
 * @f[
 *   \frac{\partial^2 \Phi}{\partial v^2}\bigg|_{\Omega_k^n}
 *   = \Lambda^{n-1}\, (P_k\, \bar A\, V)^\top
 *     \frac{\partial^2 b}{\partial v_p^2}(t_{n,k}(v, w))
 *     \cdot \big(\partial v_p / \partial v\big)^2.
 * @f]
 * The squared Jacobian eliminates the per-tile sign:
 * @f$ \sigma_k^2 = 1 @f$ for all @f$ k @f$, and the cross term
 * @f$ \sigma_k \sigma_k = 1 @f$ for the medial tile too. So every
 * second-derivative component picks up exactly @f$ 4^n @f$ — no
 * branching on @f$ k @f$.
 *
 * @section ret_hess Return shape
 *
 * @f$ K \times 3 @f$, with columns matching @ref regular_basis_hess —
 *  - column 0 = @f$ \partial^2 \Phi / \partial v^2 @f$
 *  - column 1 = @f$ \partial^2 \Phi / \partial v \partial w @f$
 *  - column 2 = @f$ \partial^2 \Phi / \partial w^2 @f$
 *
 * @section jordan_hess N = 3 Jordan block
 *
 * Same as @ref stam_phi — Jordan correction wired in via the shared
 * helper; Hessian is exact for @c N == 3 too.
 *
 * @throws std::invalid_argument on the same conditions as
 *         @ref stam_tile_map.
 */
[[nodiscard]] Eigen::Matrix<double, Eigen::Dynamic, 3>
stam_phi_hess(const StamEvaluator& ev, double v, double w);

/**
 * @brief Limit-surface quantities at one parametric point on an
 *        irregular Stam patch (mirror of @ref PatchEvaluation but with
 *        @f$ K = N + 6 @f$ DOFs instead of 12).
 *
 * Holds everything the Cirak-Ortiz strain matrices need at the given
 * quadrature point: the 3D position and tangent / second-derivative
 * basis vectors evaluated from the K patch control vertices, plus the
 * raw Stam basis values and derivatives the per-element stiffness
 * routine consumes.
 */
struct StamPatchEvaluation {
    Eigen::Vector3d                                 position;       ///< 3D position @f$ x(v, w) @f$.
    Eigen::Matrix<double, 3, 2>                     cov_basis;      ///< Covariant tangents @f$ x_{,v}, x_{,w} @f$.
    Eigen::Vector3d                                 normal;         ///< Unit surface normal @f$ a_3 @f$.
    Eigen::Matrix<double, 3, 3>                     second_derivs;  ///< @f$ x_{,vv}, x_{,vw}, x_{,ww} @f$.
    Eigen::VectorXd                                 N;              ///< Length @c K = N+6.
    Eigen::Matrix<double, Eigen::Dynamic, 2>        N_grad;         ///< @c K x 2.
    Eigen::Matrix<double, Eigen::Dynamic, 3>        N_hess;         ///< @c K x 3.
};

/**
 * @brief Evaluate the Stam-exact limit surface at @f$(v, w)@f$ on an
 *        irregular patch with valence @f$ N \ge 3 @f$.
 *
 * Gathers the @f$ K = N + 6 @f$ patch control vertices indexed by
 * @p patch_dofs from @p V, contracts them against the Stam basis
 * @ref stam_phi (and its derivatives), and packages position /
 * tangents / normal / second-derivatives in the same form as
 * @ref evaluate_patch_regular. The element-stiffness kernel
 * @ref element_stiffness_stam consumes the result.
 *
 * @param ev          Cached @ref StamEvaluator for valence @f$ N @f$.
 * @param patch_dofs  Length-@c K vertex indices in @p V indexing the
 *                    canonical Stam patch (slot 0 = extraordinary
 *                    vertex, 1..N = 1-ring, N+1..N+5 = outer 5).
 * @param V           Vertex matrix, @f$ n \times 3 @f$.
 * @param v, w        Parametric evaluation point on the standard
 *                    unit triangle.
 *
 * @return Populated @ref StamPatchEvaluation.
 *
 * @throws std::invalid_argument
 *         if @c V has fewer than 3 columns; if @c patch_dofs.size()
 *         != @c K; or if @c (v, w) is outside the unit triangle (or at
 *         the extraordinary-vertex limit).
 */
[[nodiscard]] StamPatchEvaluation evaluate_patch_stam(
    const StamEvaluator&                ev,
    const std::vector<Eigen::Index>&    patch_dofs,
    const Eigen::MatrixXd&              V,
    double                              v,
    double                              w);

/**
 * @brief Per-element stiffness matrix for an irregular Stam patch.
 *
 * Mirrors @ref element_stiffness_regular in every detail except the
 * DOF count: at each point of the reference-triangle quadrature @p rule
 * it builds membrane and bending Voigt strain-displacement matrices
 * @f$ M, B \in \mathbb{R}^{3 \times 3K} @f$ from the K Stam basis values
 * and derivatives (via @ref evaluate_patch_stam), contracts with the
 * @ref ShellMaterial elasticity, and accumulates @f$ \sum_g w_g @f$.
 *
 * The output is a @f$ 3K \times 3K @f$ symmetric element stiffness
 * matrix; rows and columns are blocks of 3 (one per spatial component),
 * @f$ K @f$ such blocks (one per Stam patch DOF in the order specified
 * by @ref evaluate_patch_stam's @c patch_dofs).
 *
 * @param ev          Cached @ref StamEvaluator for the patch's valence.
 * @param patch_dofs  Length-@c K vertex indices in @p V_aug.
 * @param V_aug       Augmented rest-vertex matrix (same as for K assembly).
 * @param material    Membrane / bending prefactors and Poisson ratio.
 * @param rule        Reference-triangle quadrature rule (default 7-point
 *                    Dunavant degree-5).
 *
 * @return @f$ 3K \times 3K @f$ element stiffness matrix.
 *
 * @throws std::runtime_error
 *         on a degenerate parametrisation (@f$|a_1 \times a_2| = 0@f$).
 */
[[nodiscard]] Eigen::MatrixXd element_stiffness_stam(
    const StamEvaluator&                ev,
    const std::vector<Eigen::Index>&    patch_dofs,
    const Eigen::MatrixXd&              V_aug,
    const chladni::shell::ShellMaterial& material,
    QuadratureRule                       rule = QuadratureRule::SevenPointDunavant);

/**
 * @brief Per-element consistent mass matrix for an irregular Stam patch.
 *
 * Mirrors @ref element_mass_regular for irregular patches. Integrates
 * @f$\rho h \int N_I N_J \sqrt{a}\,dv\,dw@f$ with 7-point Dunavant
 * degree-5 quadrature; calls @ref evaluate_patch_stam at each quadrature
 * point to obtain the Stam basis values and the surface area element
 * (the Stam evaluation is exact at any @f$(v, w)@f$ on the unit triangle
 * except the apex @f$(0, 0)@f$, where the @f$n \to \infty@f$ eigenmode
 * limit is singular — the chosen Dunavant points are all safely inside).
 *
 * Output is a @f$3K \times 3K@f$ block-diagonal matrix in the spatial
 * components (each (I, J) block is @f$M_e^{IJ} \cdot I_3@f$), one block
 * per Stam patch DOF in the order specified by @p patch_dofs.
 *
 * @param ev              Cached @ref StamEvaluator for the patch's valence.
 * @param patch_dofs      Length-@c K vertex indices.
 * @param V_aug           Augmented vertex matrix.
 * @param surface_density @f$\rho h@f$ (kg/m²), @f$> 0@f$.
 * @param rule            Reference-triangle quadrature rule (default 7-point
 *                        Dunavant degree-5).
 *
 * @return Symmetric positive-semi-definite @f$M_e \in \mathbb{R}^{3K \times 3K}@f$.
 *
 * @throws std::invalid_argument on empty @p patch_dofs.
 * @throws std::runtime_error    on a degenerate parametrisation at any
 *                               quadrature point.
 */
[[nodiscard]] Eigen::MatrixXd element_mass_stam(
    const StamEvaluator&             ev,
    const std::vector<Eigen::Index>& patch_dofs,
    const Eigen::MatrixXd&           V_aug,
    double                           surface_density,
    QuadratureRule                   rule = QuadratureRule::SevenPointDunavant);

}  // namespace chladni::shell::loop
