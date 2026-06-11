#pragma once

/**
 * @file shell.hpp
 * @brief Discrete-shells FEM building blocks for the chladni pipeline.
 *
 * The thin-shell modal solve is built up from three pieces:
 *
 * 1. Mesh geometry — face areas, an oriented edge list with the two
 *    adjacent triangles per edge, and undeformed dihedral angles
 *    (this header).
 * 2. A 3n x 3n lumped mass matrix M derived from face areas, density
 *    and thickness (this header).
 * 3. A 3n x 3n discrete-shells stiffness matrix K assembled from
 *    membrane (edge-length) and bending (dihedral-angle) energies
 *    (Stage D, lands separately).
 *
 * Solving the generalised eigenproblem @f$ K \phi = \omega^2 M \phi @f$
 * then yields the modal frequencies and shapes used by the synthesiser.
 *
 * @section shell_refs References
 * - @cite grinspun_2003_discrete_shells — discrete-shells formulation
 *   (provides geometric framework: per-edge length / per-hinge bending).
 * - @cite wardetzky_2007_quadratic_curvature — IBM bending energy
 *   (per-hinge cotangent-weighted rank-1 quadratic with constant Hessian).
 *
 * @section formulation Formulation choices
 *
 * The membrane energy uses the **constant strain triangle** (CST) plane-
 * stress FEM element: each triangle contributes a 9x9 stiffness
 * @f$ K_T = A_T\,B^\top D B @f$ where @f$D = (E h / (1-\nu^2)) [...] @f$
 * is the plane-stress isotropic stiffness. This replaces the per-edge
 * spring @f$ k_L (L_e - \bar L_e)^2 / \bar L_e @f$ used in earlier
 * versions; CST faithfully represents in-plane biaxial deformation with
 * arbitrary Poisson ratio.
 *
 * The bending energy uses the **Wardetzky 2007 quadratic curvature
 * model** (IBM): each interior edge with hinge stencil
 * @f$(v_0, v_1, c_L, c_R)@f$ contributes a 12x12 stiffness
 * @f$ K_e = (3 D_b / A_0)\, c c^\top \otimes I_3 @f$ with
 * @f$D_b = E h^3 / (12 (1-\nu^2))@f$, @f$A_0@f$ the combined rest area
 * of the two adjacent triangles, and @f$c \in \mathbb{R}^4@f$ the
 * cotangent-weighted vertex coefficients. This replaces the dihedral-
 * angle bending @f$ k_B (\theta_e - \bar\theta_e)^2 \bar L_e / \bar h_e @f$;
 * the IBM Hessian is constant in vertex positions (no analytic Hessian
 * approximation needed) and gives genuine convergence to the continuum
 * thin-plate bending energy under mesh refinement on curved rest meshes.
 */

#include <chladni/material.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <vector>

namespace chladni::shell {

/**
 * @brief One oriented edge of a triangle mesh and its adjacent faces.
 *
 * Endpoints @ref v0 and @ref v1 are stored with @f$ v_0 < v_1 @f$ to give
 * each edge a canonical key. The two adjacent triangle indices are
 * disambiguated by winding: @ref face_left contains the directed edge
 * @f$ v_0 \to v_1 @f$, @ref face_right contains @f$ v_1 \to v_0 @f$.
 *
 * For a closed manifold mesh both adjacent faces are populated. For a
 * boundary edge (e.g. an open shell with no end caps) exactly one of
 * the two is -1 — which one depends on the boundary face's winding.
 *
 * @see build_edges()
 */
struct Edge {
    Eigen::Index v0;          ///< first endpoint, with @f$ v_0 < v_1 @f$.
    Eigen::Index v1;          ///< second endpoint.
    Eigen::Index face_left;   ///< face that has @f$ v_0 \to v_1 @f$ in its winding,
                              ///< or -1 if no such face exists.
    Eigen::Index face_right;  ///< face that has @f$ v_1 \to v_0 @f$ in its winding,
                              ///< or -1 if no such face exists.

    /// True iff both adjacent faces are present.
    [[nodiscard]] bool is_interior() const noexcept
    {
        return face_left != -1 && face_right != -1;
    }
    /// True iff exactly one adjacent face is missing (manifold boundary edge).
    [[nodiscard]] bool is_boundary() const noexcept { return !is_interior(); }
};

/**
 * @brief Compute the area of every triangle in the mesh.
 *
 * @param V  Vertex matrix, @f$ n \times 3 @f$.
 * @param F  Face index matrix, @f$ m \times 3 @f$.
 *
 * @return Vector of length @f$ m @f$, each entry the unsigned 3D area
 *         @f$ A_f = \tfrac{1}{2} \| (v_1 - v_0) \times (v_2 - v_0) \| @f$
 *         of triangle @f$ f @f$.
 */
Eigen::VectorXd face_areas(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F);

/**
 * @brief Build the oriented edge list of a triangle mesh.
 *
 * @param F  Face index matrix, @f$ m \times 3 @f$.
 *
 * @return Vector of @ref Edge entries, sorted lexicographically by
 *         @c (v0, v1). For a manifold input every interior edge has
 *         exactly one left and one right face; boundary edges have
 *         only @ref Edge::face_left set.
 *
 * @throws std::runtime_error if a non-manifold edge is detected
 *         (more than two adjacent faces, or the same edge orientation
 *         appears twice).
 */
std::vector<Edge> build_edges(const Eigen::MatrixXi& F);

/**
 * @brief Per-edge geometric data measured in the rest configuration.
 *
 * Filled in by @ref compute_edge_rest_data() once per shell. The
 * fields are everything the discrete-shells energy and its gradient
 * need at evaluation time.
 *
 * @section conventions Sign and orientation
 * The bend angle @ref dihedral is the **signed** rotation that takes
 * the left face's normal to the right face's normal about the
 * directed edge @f$ v_0 \to v_1 @f$:
 * @f[
 *   \theta_e \;=\; \mathrm{atan2}\bigl((n_L \times n_R) \cdot \hat e,\;
 *                                       n_L \cdot n_R\bigr)
 * @f]
 * with @f$ \hat e @f$ the unit edge vector and @f$ n_L, n_R @f$ the
 * outward unit face normals. With this convention a flat shell has
 * @f$ \theta_e = 0 @f$ everywhere; a cylinder with consistent
 * outward normals has @f$ \theta_e \approx 2\pi/N_\text{around} @f$
 * around its rim per panel.
 *
 * For boundary edges (@ref Edge::is_boundary returns true) the
 * dihedral and wing fields are not defined and are filled with
 * sentinels: @ref dihedral = 0, @ref c_left and @ref c_right = -1.
 */
struct EdgeRestData {
    double          length;     ///< Rest edge length @f$ \bar L_e = \|v_1 - v_0\| @f$.
    double          dihedral;   ///< Rest signed bend angle @f$ \bar\theta_e @f$ (rad).
                                ///< Diagnostic only after the Wardetzky 2007 IBM
                                ///< rewrite; bending energy no longer uses dihedrals.
    double          h_e;        ///< @f$\tfrac{1}{3}@f$ * mean of the perpendicular
                                ///< heights of the two incident faces to the edge,
                                ///< per @cite grinspun_2003_discrete_shells eq. (2).
                                ///< Diagnostic only after the IBM rewrite.
    Eigen::Vector3d hat;        ///< Unit edge direction in rest config, @f$ v_0 \to v_1 @f$.
    Eigen::Index    c_left;     ///< Index of the third vertex of @c face_left, or -1 if boundary.
    Eigen::Index    c_right;    ///< Index of the third vertex of @c face_right, or -1 if boundary.

    /// Combined rest area of the two adjacent triangles, @f$ A_0 = A_L + A_R @f$,
    /// used as the denominator of the IBM bending energy. Set to 0 on boundary edges.
    double          combined_area;
    /// Wardetzky 2007 IBM cotangent coefficients @f$(c_{v_0}, c_{v_1}, c_{c_L}, c_{c_R})@f$
    /// in the same vertex order as the 12-vector concatenation
    /// @f$(x_{v_0}, x_{v_1}, x_{c_L}, x_{c_R})@f$. Sum of components is zero
    /// (translation invariance). Filled with zeros on boundary edges.
    /// See @cite wardetzky_2007_quadratic_curvature theorem 3 / eq. (8).
    Eigen::Vector4d c_ibm;
};

/**
 * @brief Compute @ref EdgeRestData for every edge of the shell.
 *
 * @param V      Vertex matrix, @f$ n \times 3 @f$, in the rest configuration.
 * @param F      Face index matrix, @f$ m \times 3 @f$.
 * @param edges  Edge list returned by @ref build_edges (must be the same
 *               edges returned for @p F).
 *
 * @return Length-`edges.size()` vector of rest data entries, in the
 *         same order as @p edges.
 *
 * @throws std::runtime_error
 *         on a degenerate face (zero area) — the dihedral angle is
 *         undefined when an adjacent face has no normal.
 */
std::vector<EdgeRestData> compute_edge_rest_data(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges);

/**
 * @brief Plane-stress shell stiffness coefficients.
 *
 * Maps the user-facing isotropic linear-elastic material onto the
 * coefficients used by the CST membrane and Wardetzky IBM bending
 * formulations:
 *
 *  - @ref k_L  is the plane-stress membrane prefactor
 *              @f$ E h / (1 - \nu^2) @f$ (units: N/m = kg/s^2).
 *              Multiplies the constant-strain-triangle stiffness
 *              @f$ K_T = A_T B^\top D B @f$ where the 3x3 plane-stress
 *              elasticity matrix is
 *              @f$ D = k_L \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & (1-\nu)/2 \end{bmatrix}.@f$
 *              Setting @c k_L = 0 disables the membrane contribution.
 *  - @ref k_B  is the plate flexural rigidity
 *              @f$ D_b = E h^3 / (12 (1 - \nu^2)) @f$ (units: J = N*m).
 *              Multiplies the IBM per-hinge stiffness
 *              @f$ K_e = (3 D_b / A_0) c c^\top \otimes I_3 @f$. Setting
 *              @c k_B = 0 disables the bending contribution.
 *  - @ref poisson_ratio is @f$\nu@f$, used both for the off-diagonals
 *              of @f$D@f$ and (already folded into @c k_L, @c k_B) for
 *              the prefactors. Required because @f$D@f$ has a non-trivial
 *              structure that depends on @f$\nu@f$ separately from the
 *              overall scaling.
 *
 * @see shell_material_from_isotropic
 */
struct ShellMaterial {
    double k_L;             ///< membrane prefactor @f$E h / (1-\nu^2)@f$ (N/m).
    double k_B;             ///< bending prefactor @f$E h^3 / (12 (1-\nu^2))@f$ (J).
    double poisson_ratio;   ///< Poisson ratio @f$\nu@f$, used inside the CST D-matrix.
};

/**
 * @brief Calibrate @ref ShellMaterial from a physical isotropic material.
 *
 * Computes the two textbook plane-stress prefactors:
 *  - @f$ k_L = E h / (1 - \nu^2) @f$ — membrane CST scaling.
 *  - @f$ k_B = E h^3 / (12 (1 - \nu^2)) @f$ — plate flexural rigidity.
 *  - @f$ \nu @f$ stored verbatim for the CST D-matrix structure.
 *
 * @throws std::invalid_argument if @p thickness is non-positive,
 *         Young's modulus is non-positive, or @c nu is outside
 *         @f$(-1, 1/2)@f$.
 */
[[nodiscard]] ShellMaterial shell_material_from_isotropic(
    const ::chladni::IsotropicMaterial& mat,
    double thickness);

/**
 * @brief Linearised CST + Wardetzky IBM elastic energy in displacement form.
 *
 * The energy is a quadratic form in the displacement
 * @f$ u = V_\text{current} - V_\text{rest} @f$, written as a sum over
 * triangles (membrane) and a sum over interior edges (bending):
 * @f[
 *   W(u) \;=\;
 *     \sum_T \tfrac{1}{2}\, A_T\, k_L \,\varepsilon_T(u)^\top
 *           \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & (1-\nu)/2 \end{bmatrix}
 *           \varepsilon_T(u)
 *   \;+\;
 *     \sum_{e \in \text{interior}}
 *       \dfrac{3 k_B}{2 A_{0,e}} \,\Bigl\|\sum_p c_{e,p}\, u_{e,p}\Bigr\|^2.
 * @f]
 * The strain @f$\varepsilon_T(u)@f$ is the constant-strain-triangle plane-
 * stress strain measured in the rest local 2D frame; the cotangent
 * coefficients @f$c_{e,p}@f$ and combined area @f$A_{0,e}@f$ are
 * precomputed in @ref compute_edge_rest_data.
 *
 * Properties:
 *  - @f$ W(0) = 0 @f$ exactly (energy zero at rest).
 *  - @f$ \nabla W(0) = 0 @f$ exactly (rest is a stationary point).
 *  - @f$ \nabla^2 W(0) @f$ is the stiffness assembled by
 *    @ref assemble_stiffness_at_rest_analytic, so FD second derivatives
 *    of @c shell_energy must agree with the analytic Hessian within FD
 *    truncation. This is the basis of the regression test in
 *    @c tests/shell/test_stiffness.cpp.
 *  - Translation-invariant exactly (the membrane sum has zero strain
 *    on a constant displacement, and the IBM coefficients sum to zero
 *    so the cotangent-weighted sum is zero on a uniform translation).
 *  - Rotation-invariant only to first order in the displacement; this is
 *    the standard small-strain / linearised-bending limit and is the
 *    correct setting for modal analysis at rest.
 *
 * @param V_rest     Rest vertex positions (the @c rest_data must have
 *                   been computed against this configuration).
 * @param V_current  Current vertex positions, n x 3.
 * @param F          Face indices, m x 3.
 * @param edges      Edge list (must match @p F).
 * @param rest_data  Per-edge rest data (must match @p edges and @p V_rest).
 * @param material   Stiffness coefficients.
 *
 * @return Total linearised elastic energy (Joules).
 */
[[nodiscard]] double shell_energy(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXd& V_current,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material);

/**
 * @brief Assemble the shell stiffness matrix @f$K@f$ at the rest config.
 *
 * Computes @f$ K_{ij} = \partial^2 W / \partial x_i \partial x_j @f$
 * evaluated at the rest configuration via sparsity-aware central finite
 * differencing of the @ref shell_energy functional:
 * @f[
 *   K_{ii} = \dfrac{W(x_\text{rest} + \epsilon e_i) + W(x_\text{rest} - \epsilon e_i)}{\epsilon^2},
 * @f]
 * @f[
 *   K_{ij} = \dfrac{W(+\epsilon e_i + \epsilon e_j) + W(-\epsilon e_i - \epsilon e_j)
 *                  - W(+\epsilon e_i - \epsilon e_j) - W(-\epsilon e_i + \epsilon e_j)}
 *                 {4\,\epsilon^2}, \quad i \neq j.
 * @f]
 * (W vanishes exactly at the rest configuration so the diagonal form
 * has no @f$ -2 W(0) @f$ subtraction.)
 *
 * Sparsity: the stencil over which W has any dependence on a vertex
 * pair @f$(a, b)@f$ is exactly the union over edges of:
 *  - membrane edges  @f$\{v_0, v_1\}@f$
 *  - interior edges  @f$\{v_0, v_1, c_\text{left}, c_\text{right}\}@f$
 * so we only evaluate FD blocks for vertex pairs in this stencil set.
 *
 * @param V_rest    rest vertex positions
 * @param F         face indices
 * @param edges     edge list (must match @p F)
 * @param rest_data rest data (must match @p edges)
 * @param material  shell stiffness coefficients
 * @param eps       FD step (1e-6 by default; double precision optimum
 *                  for second derivatives is around 1e-4 to 1e-6
 *                  depending on energy magnitude)
 *
 * @return @f$ 3n \times 3n @f$ sparse symmetric stiffness matrix.
 *
 * @note Cost is O(|stencil_pairs| * 9) energy evaluations. For a
 *       cylinder.obj-sized mesh (~hundreds of vertices, ~thousands
 *       of stencil pairs) this is well under a second.
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_stiffness_at_rest_fd(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material,
    double eps = 1.0e-6);

/**
 * @brief Assemble the shell stiffness matrix @f$K@f$ at the rest config
 * by direct evaluation of the closed-form CST + IBM Hessian.
 *
 * The shell energy is a sum of per-element quadratic forms whose
 * Hessians are constant in vertex positions; this routine evaluates
 * those Hessians directly without any finite-difference step.
 *
 * **Membrane (CST) per triangle.**
 * For each triangle @f$T@f$ with rest area @f$A_T@f$ and rest local 2D
 * frame @f$(\hat t_1, \hat t_2)@f$:
 *   - Map each vertex's 3D displacement onto the local 2D frame to give
 *     the projection matrix @f$P \in \mathbb{R}^{6\times 9}@f$.
 *   - The 2D strain-displacement matrix
 *     @f$B_{2D} \in \mathbb{R}^{3\times 6}@f$ is the standard CST form.
 *   - The local 3x3 plane-stress stiffness is
 *     @f$D = k_L \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & (1-\nu)/2 \end{bmatrix}@f$.
 *   - The 9x9 element stiffness scattered into @c K is
 *     @f$ K_T = A_T (B_{2D} P)^\top D (B_{2D} P) @f$.
 *
 * **Bending (Wardetzky IBM) per interior edge.**
 * For each interior edge with hinge stencil @f$(v_0, v_1, c_L, c_R)@f$,
 * combined rest area @f$A_0@f$, and cotangent coefficients
 * @f$c \in \mathbb{R}^4@f$ (precomputed in @ref EdgeRestData::c_ibm):
 *   @f[
 *     K_e \;=\; \dfrac{3 k_B}{A_0}\, c\, c^\top \otimes I_3
 *   @f]
 * where @f$\otimes I_3@f$ replicates each scalar @f$c_p c_q@f$ on the
 * 3x3 diagonal of the corresponding vertex-pair block. See
 * @cite wardetzky_2007_quadratic_curvature theorem 3.
 *
 * Both formulations have constant Hessians in vertex positions, so the
 * matrix is exactly the second derivative of the linearised
 * @ref shell_energy regardless of the mesh's rest curvature.
 *
 * @param V_rest    rest vertex positions
 * @param F         face indices
 * @param edges     edge list (must match @p F)
 * @param rest_data rest data (must match @p edges); the cotangent
 *                  coefficients and combined area must be filled in.
 * @param material  shell stiffness coefficients
 *
 * @return @f$ 3n \times 3n @f$ sparse symmetric stiffness matrix.
 *         Entries equal those of @ref assemble_stiffness_at_rest_fd
 *         within FD truncation error (~1e-4 relative on cylinder.obj).
 */
[[nodiscard]] Eigen::SparseMatrix<double> assemble_stiffness_at_rest_analytic(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material);

/**
 * @brief Lumped per-vertex mass for a thin shell.
 *
 * Each vertex receives @f$ \tfrac{1}{3} @f$ of the mass of every
 * adjacent face: @f$ m_i = \rho h \sum_{f \ni i} \tfrac{A_f}{3} @f$.
 * The sum @f$ \sum_i m_i @f$ equals the total shell mass
 * @f$ \rho h \cdot \mathrm{Area}(\Sigma) @f$ exactly.
 *
 * @param V          Vertex matrix, @f$ n \times 3 @f$.
 * @param F          Face index matrix, @f$ m \times 3 @f$.
 * @param density    Mass density @f$\rho@f$ (kg/m^3); @f$ > 0 @f$.
 * @param thickness  Shell thickness @f$h@f$ (m); @f$ > 0 @f$.
 *
 * @return Length-@f$n@f$ vector of per-vertex scalar masses (kg).
 *
 * @throws std::invalid_argument if @p density or @p thickness is non-positive.
 */
Eigen::VectorXd lumped_vertex_masses(const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     double density,
                                     double thickness);

/**
 * @brief A computed set of shell vibration modes.
 *
 * Returned by @ref compute_shell_modes. Modes are sorted in ascending
 * order of frequency; rigid-body zero modes are filtered out.
 */
struct ShellModes {
    Eigen::VectorXd omegas;  ///< Angular frequencies @f$\omega_i@f$ (rad/s),
                             ///< length @c n_modes, ascending.
    Eigen::MatrixXd shapes;  ///< Mode shapes, @f$3n \times n_\text{modes}@f$;
                             ///< column @c k is the displacement field of
                             ///< mode @c k. Mass-orthonormalised:
                             ///< @f$ \phi_i^T M \phi_j = \delta_{ij} @f$.
};

/**
 * @brief Compute the lowest natural modes of a thin shell.
 *
 * Wraps the full pipeline:
 *  -# build the edge list and per-edge rest data,
 *  -# assemble the lumped mass matrix @f$M@f$,
 *  -# assemble the stiffness matrix @f$K@f$ from the Wardetzky 2007 IBM
 *     bending energy via its constant Hessian @f$ c c^\top \otimes I_3 @f$
 *     (see @cite wardetzky_2007_quadratic_curvature; the per-hinge stencil
 *     detailed in the file header above),
 *  -# solve the generalised eigenproblem @f$ K \phi = \omega^2 M \phi @f$
 *     for the smallest @c n_modes + slack eigenvalues using Spectra in
 *     multi-seed shift-invert mode,
 *  -# discard the (up to 6) near-zero rigid-body modes and return the
 *     next @c n_modes physical modes.
 *
 * @note This is the legacy CST + isometric-bending path. The meshfree
 *       (LME/SME) and Loop assemblers reached through @ref ShellAssembler
 *       use a consistent mass matrix; the eigensolver (shift-invert) is
 *       shared by all paths.
 *
 * @param V          Vertex matrix in the rest configuration.
 * @param F          Face indices.
 * @param material   Isotropic linear-elastic material.
 * @param thickness  Shell thickness @f$h@f$ (m), @f$> 0@f$.
 * @param n_modes    Number of physical modes to return.
 *
 * @return ShellModes with @c n_modes ascending angular frequencies and
 *         their corresponding displacement shapes.
 *
 * @throws std::invalid_argument
 *         on non-positive thickness or zero @c n_modes.
 * @throws std::runtime_error
 *         on Spectra non-convergence or insufficient physical modes.
 *
 * @note Shell modes are sensitive to boundary conditions. The returned
 *       spectrum is for an unconstrained (free-free) shell — the
 *       physically correct case for a drum body. To match closed-form
 *       references that assume tangentially-fixed (SD-SD) edges, the
 *       caller must constrain the boundary DOFs externally.
 */
[[nodiscard]] ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    double thickness,
    std::size_t n_modes);

/**
 * @brief Same as @ref compute_shell_modes but with explicit @ref ShellMaterial.
 *
 * Lets callers override the stiffness coefficients independently of
 * the isotropic material — useful for diagnostic tests that want to
 * disable membrane (`sm.k_L = 0`) or bending (`sm.k_B = 0`) selectively.
 *
 * @param V          Vertex matrix in the rest configuration.
 * @param F          Face indices.
 * @param material   Used only for `density` (mass-matrix assembly).
 * @param sm         Stiffness coefficients used to assemble @c K.
 * @param thickness  Shell thickness @f$h@f$ (m), @f$> 0@f$.
 * @param n_modes    Number of physical modes to return.
 */
[[nodiscard]] ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double thickness,
    std::size_t n_modes);

/**
 * @brief Loop-subdivision shell modal solve.
 *
 * Companion to @ref compute_shell_modes that swaps the CST + Wardetzky-IBM
 * stiffness assembly for a Cirak-Ortiz-Schroder 2000 Loop-subdivision
 * thin-shell formulation (@ref chladni::shell::loop::assemble_stiffness_loop).
 * Mass matrix, eigensolver path (Spectra shift-invert), and rigid-body
 * filter (mass-orthogonal projection against translations + rotations)
 * are identical to @ref compute_shell_modes; only @f$K@f$ is rebuilt
 * with strict @f$C^1@f$ box-spline elements.
 *
 * @section loop_solver_status Implementation status
 *
 * The Loop assembly handles:
 *  - **Regular interior patches** (every corner valence = 6) — direct
 *    quartic box-spline basis.
 *  - **Boundary vertices** of valence 2, 3, and 4 — Schweitzer
 *    phantom-vertex augmentation (@ref chladni::shell::loop::augment_for_loop_boundary).
 *    Covers cylinder rims (val-4), rectangular plate corners (val-3),
 *    strip endpoints (val-2). Boundary valence > 4 (e.g. T-junctions)
 *    still throws.
 *  - **Extraordinary interior vertices** (valence != 6) via L.3.4
 *    step-1 multi-pass approximation (default) or the S-series Stam
 *    exact eigenbasis evaluation (opt-in via @p use_stam).
 *
 * Every bundled mesh in @c models/ is handled by the Loop path —
 * verified by the @c [.smoke] bundled-mesh test, which sweeps the
 * directory.
 *
 * @section motivation Motivation
 *
 * The legacy CST + IBM assembly is over-stiff by ~60% on the absolute-
 * frequency calibration tests against the Euler-Bernoulli beam and the
 * Rayleigh-Love free-free cylindrical-shell reference (see
 * @c test_modes_vs_free_free_cylinder_analytic.cpp). Cirak-Ortiz Loop
 * elements give a strict @f$C^1@f$ basis without rotational DOFs, which
 * removes the locking and direction-dependence that the IBM bending
 * Hessian exhibits on regular triangulations.
 *
 * @param V          Vertex matrix in the rest configuration.
 * @param F          Face indices.
 * @param material   Isotropic linear-elastic material.
 * @param thickness  Shell thickness @f$h@f$ (m), @f$> 0@f$.
 * @param n_modes    Number of physical modes to return.
 * @param n_passes   Number of subdivision passes used in the L.3.4
 *                   irregular branch (forwarded to
 *                   @ref chladni::shell::loop::assemble_stiffness_loop).
 * @param use_stam   Selects the S-series Stam exact-evaluation path for
 *                   irregular sub-triangles instead of the L.3.4 step-1
 *                   drop approximation.
 *
 * @return ShellModes with @c n_modes ascending angular frequencies and
 *         their corresponding displacement shapes.
 *
 * @throws std::invalid_argument
 *         on non-positive thickness, zero @c n_modes, or a mesh with an
 *         unsupported boundary or extraordinary vertex (see status).
 * @throws std::runtime_error
 *         on Spectra non-convergence or insufficient physical modes.
 */
[[nodiscard]] ShellModes compute_shell_modes_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    double      thickness,
    std::size_t n_modes,
    int         n_passes = 1,
    bool        use_stam = false);

/**
 * @brief Same as @ref compute_shell_modes_loop but with explicit @ref ShellMaterial.
 *
 * Lets callers override the stiffness coefficients independently of the
 * isotropic material — useful for diagnostic tests that want to disable
 * membrane (@c sm.k_L = 0) or bending (@c sm.k_B = 0) selectively.
 *
 * @param V          Vertex matrix in the rest configuration.
 * @param F          Face indices.
 * @param material   Used only for @c density (mass-matrix assembly).
 * @param thickness  Shell thickness @f$h@f$ (m), @f$> 0@f$.
 * @param n_modes    Number of physical modes to return.
 * @param sm         Stiffness coefficients used to assemble @c K via
 *                   @ref chladni::shell::loop::assemble_stiffness_loop.
 * @param n_passes  Number of subdivision passes used in the L.3.4
 *                  irregular branch (forwarded to
 *                  @ref chladni::shell::loop::assemble_stiffness_loop).
 * @param use_stam  Forwarded to
 *                  @ref chladni::shell::loop::assemble_stiffness_loop;
 *                  selects the S-series Stam exact-evaluation path for
 *                  irregular sub-triangles (instead of the L.3.4 step-1
 *                  drop approximation).
 */
[[nodiscard]] ShellModes compute_shell_modes_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double      thickness,
    std::size_t n_modes,
    int         n_passes = 1,
    bool        use_stam = false);

class ShellAssembler;  // include chladni/shell/assembler.hpp at the call site.

/**
 * @brief Modal solve through a polymorphic @ref ShellAssembler.
 *
 * Framework-aware entry point: the caller picks a concrete assembler
 * (e.g. @ref LoopAssembler with explicit
 * @ref LoopAssembler::Params) and this function pulls @c K and @c M
 * from it, then runs the standard rigid-body-filtered eigensolve. Used
 * by the GUI tab-bar and A/B numerical experiments where the assembler
 * configuration is the variable under study.
 *
 * Internally identical to @ref compute_shell_modes_loop except that
 * matrix assembly is delegated to @p assembler — including the
 * post-assembly mass-lumping policy (the assembler returns the final
 * @c M, lumped if its Params say so).
 *
 * @param V          Vertex matrix in the rest configuration.
 * @param F          Face indices.
 * @param material   Isotropic linear-elastic material. Used only for
 *                   @c density (mass-matrix assembly).
 * @param sm         Stiffness coefficients passed to
 *                   @ref ShellAssembler::assemble_K.
 * @param thickness  Shell thickness @f$ h @f$ (m), @f$ > 0 @f$.
 * @param n_modes    Number of physical modes to return, @f$ \ge 1 @f$.
 * @param assembler  Active formulation. Must outlive the call.
 *
 * @return Ascending angular frequencies + displacement shapes.
 *
 * @throws std::invalid_argument on non-positive thickness or zero
 *         @c n_modes.
 * @throws std::runtime_error on Spectra non-convergence or insufficient
 *         physical modes (forwarded from the rigid-body-filtered solver).
 */
[[nodiscard]] ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double      thickness,
    std::size_t n_modes,
    const ShellAssembler& assembler);

/**
 * @brief Build the @f$ 3n \times 3n @f$ sparse diagonal mass matrix.
 *
 * Each vertex contributes its scalar lumped mass on three diagonal
 * entries (one per spatial component @f$ x, y, z @f$); all
 * off-diagonal entries are zero. The matrix is therefore strictly
 * positive-definite and inversion is trivial.
 *
 * @param vertex_masses Length-@f$n@f$ vector of per-vertex masses, e.g.
 *                      from @ref lumped_vertex_masses.
 *
 * @return Sparse @f$ 3n \times 3n @f$ matrix in compressed form, with
 *         exactly @f$ 3n @f$ non-zeros on the diagonal.
 */
Eigen::SparseMatrix<double>
assemble_mass_matrix(const Eigen::VectorXd& vertex_masses);

}  // namespace chladni::shell
