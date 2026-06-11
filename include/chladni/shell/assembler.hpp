#pragma once

/**
 * @file assembler.hpp
 * @brief Abstract shell-FEM assembler interface for formulation comparison.
 *
 * Defines a thin polymorphic interface at the assembled-matrix level:
 * a @ref chladni::shell::ShellAssembler can produce a stiffness matrix
 * @f$ K @f$ and a consistent (or lumped) mass matrix @f$ M @f$ for the
 * generalized eigenproblem @f$ K v = \omega^2 M v @f$ on a given mesh.
 *
 * The concrete subclass @ref chladni::shell::LoopAssembler wraps the
 * existing Cirak-Ortiz-Schroder 2000 Loop-subdivision FEM (see
 * @ref chladni::shell::loop). Its @ref chladni::shell::LoopAssembler::Params struct
 * exposes the per-formulation knobs: K/M quadrature rule, mass-lumping
 * policy, subdivision passes, Stam exact-evaluation toggle. Future
 * concrete classes (e.g. WardetzkyAssembler) plug into the same
 * interface without touching `compute_shell_modes`, the GUI, or
 * downstream tests.
 *
 * Design notes (see project memory
 * `project_chladni_assembler_interface_plan` for the full rationale):
 *  - The interface lives at the **assembled matrix** level, not at
 *    per-element kernels. Different formulations (Loop basis vs.
 *    dihedral-hinge bending) have incompatible element notions, so a
 *    finer-grained interface would leak.
 *  - No formulation composition. A "Loop K + Wardetzky M" combo would
 *    need its own subclass — keeping each `ShellAssembler` internally
 *    consistent avoids spurious modes from K/M test-function mismatch.
 *  - Params are public on each concrete class. Tests and the GUI build
 *    explicit configurations like
 *    `LoopAssembler{LoopAssembler::Params{.k_quad = OnePointCentroid,
 *    .m_lump = MassLumping::None}}` for paper-exact baselines or A/B
 *    comparisons.
 *
 * @see chladni::shell::LoopAssembler
 * @see chladni::shell::compute_shell_modes (the eigensolver entry point
 *      that consumes a `const ShellAssembler&`)
 */

#include <chladni/shell.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <memory>
#include <string>
#include <vector>

namespace chladni::shell {

/**
 * @brief Triangle-reference quadrature rule for element integrals.
 *
 * Selects how the per-element @f$ K @f$ and @f$ M @f$ integrals are
 * sampled on the reference unit-area triangle @f$ \{(v,w) : v,w \ge 0,
 * v + w \le 1\} @f$ when assembling on a Loop limit surface.
 *
 * The Cirak-Ortiz-Schroder 2000 statics paper (Section 4.6) recommends
 * the @ref QuadratureRule::OnePointCentroid rule for plate-bending problems with the
 * 12-DOF regular box-spline basis; modal analysis on the same basis is
 * better served by higher-order rules. The current shipped default is
 * @ref QuadratureRule::SevenPointDunavant; user-driven A/B against the centroid rule is
 * one of the first uses of this framework.
 */
enum class QuadratureRule {
    /// Centroid rule, exact for polynomials of degree @f$ \le 1 @f$.
    /// Single sample at @f$ (v,w) = (1/3, 1/3) @f$, weight @f$ 1/2 @f$.
    OnePointCentroid,

    /// Three-point edge-midpoint rule, exact for degree @f$ \le 2 @f$.
    /// Samples at edge midpoints, each with weight @f$ 1/6 @f$.
    ThreePointEdgeMid,

    /// Dunavant degree-5 rule (7 sample points). Exact for polynomials
    /// of total degree @f$ \le 5 @f$ on the reference triangle. This is
    /// the Loop module's shipped default for both K and M integrals.
    SevenPointDunavant,

    /// Dunavant degree-6 rule (12 sample points). Exact for polynomials
    /// of total degree @f$ \le 6 @f$. This is the rule the max-ent
    /// curved-shell paper uses (Millán 2011 §4.1.1: "a standard
    /// Gauss–Legendre cubature rule of 12 points (order 6) per triangle").
    TwelvePointDunavant
};

/**
 * @brief One quadrature sample on the reference unit-area triangle.
 *
 * The reference domain is the triangle
 * @f$ T = \{(v, w) : v \ge 0, w \ge 0, v + w \le 1\} @f$ of area
 * @f$ 1/2 @f$. Weights returned by @ref quadrature_points are
 * normalised so that @f$ \sum_q \text{weight}_q = 1/2 @f$ — i.e. they
 * integrate the constant function @c f=1 to the area of @f$ T @f$.
 */
struct QuadraturePoint {
    double v;       ///< First barycentric coordinate, @f$ v \in [0, 1] @f$.
    double w;       ///< Second barycentric coordinate, @f$ w \in [0, 1-v] @f$.
    double weight;  ///< Quadrature weight (sum over the rule equals @f$ 1/2 @f$).
};

/**
 * @brief Sample points + weights for a reference-triangle quadrature rule.
 *
 * Returns the canonical sample set for @p rule. Used by the Loop
 * element kernels (and any future formulation that integrates over the
 * unit-area reference triangle):
 *
 *  - @ref QuadratureRule::OnePointCentroid — 1 sample at @f$ (1/3, 1/3) @f$.
 *  - @ref QuadratureRule::ThreePointEdgeMid — 3 samples at the edge midpoints.
 *  - @ref QuadratureRule::SevenPointDunavant — Dunavant degree-5 (7 samples).
 *
 * The total weight per rule is exactly @f$ 1/2 @f$ (the area of the
 * reference unit-area triangle); each rule integrates polynomials of
 * total degree @f$ \le 1 @f$, @f$ \le 2 @f$, and @f$ \le 5 @f$
 * exactly, respectively.
 */
[[nodiscard]] std::vector<QuadraturePoint> quadrature_points(QuadratureRule rule);

/**
 * @brief Mass-matrix lumping policy.
 *
 * Applied to the fully consistent mass matrix coming out of element
 * integration. Lumping replaces the full @f$ M @f$ with a diagonal
 * approximation, trading exactness for sparsity and (in the Loop
 * formulation) cleaner high-@f$m@f$ doublet mode shapes — the
 * row-sum-lumped variant empirically suppresses small off-diagonal
 * coupling that distorts zero-crossings near nodal lines.
 *
 * Row-sum lumping preserves the partition of unity: total mass equals
 * @f$ \rho h\,\mathrm{area} @f$ exactly.
 */
enum class MassLumping {
    /// Keep the fully consistent (non-diagonal) mass matrix.
    None,

    /// Row-sum lump: @f$ M_{ii}^{\text{diag}} = \sum_j M_{ij} @f$,
    /// off-diagonal entries dropped.
    RowSum
};

/**
 * @brief Abstract base class for shell-FEM matrix assemblers.
 *
 * Each concrete subclass wraps one formulation (Loop subdivision,
 * dihedral-hinge, etc.) and exposes two assembly entry points used by
 * @ref compute_shell_modes (the modal-solve entry point):
 *
 *  - @ref assemble_K — global stiffness on the input DOF layout
 *    (@f$ 3 n_\text{V} @f$ entries), reading material constants from
 *    a @ref ShellMaterial calibration.
 *  - @ref assemble_M — global mass matrix on the same DOF layout,
 *    parameterised by surface density @f$ \rho h @f$. The returned
 *    matrix already incorporates the per-formulation
 *    @ref MassLumping policy.
 *
 * Both calls must produce matrices on the *same* discrete function
 * space; assembling K and M from different spaces creates spurious
 * modes. Subclasses enforce this internally and document any caveats.
 *
 * A short @ref label is used by the GUI tab bar and diagnostic logs.
 */
class ShellAssembler {
public:
    virtual ~ShellAssembler() = default;

    /// Assemble the global stiffness matrix @f$ K @f$ on the input DOF
    /// layout. Symmetric positive-semi-definite; rigid translations
    /// annihilate @f$ K @f$ exactly.
    [[nodiscard]] virtual Eigen::SparseMatrix<double> assemble_K(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const ShellMaterial&   material) const = 0;

    /// Assemble the global mass matrix @f$ M @f$ on the input DOF
    /// layout, with the subclass's @ref MassLumping policy applied.
    /// Symmetric positive-(semi-)definite.
    [[nodiscard]] virtual Eigen::SparseMatrix<double> assemble_M(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        double                 surface_density) const = 0;

    /// Short human-readable name for GUI tabs / diagnostic output.
    /// Example: `"Loop (Cirak-Ortiz)"`, `"Wardetzky 2007"`.
    [[nodiscard]] virtual std::string label() const = 0;

    /**
     * @brief Project assembled-space eigenvectors into vertex-displacement
     *        coordinates suitable for visualization.
     *
     * Eigenvectors of @f$ K \alpha = \lambda M \alpha @f$ live in the
     * assembled basis's coefficient space. Whether those coefficients
     * coincide with vertex displacements depends on the formulation:
     *
     *  - **Interpolating bases** (Loop subdivision at smooth vertices,
     *    CST, etc.): @f$ N_i(x_j) = \delta_{ij} @f$, so the coefficient
     *    at node i *is* the displacement at vertex i. The default
     *    implementation returns @p mode_coefficients unchanged.
     *
     *  - **Non-interpolating bases** (LME meshfree, partition-of-unity
     *    EFG): the basis at vertex @f$ x_j @f$ has support on several
     *    neighbour coefficients, so the exact vertex displacement is
     *    @f$ u_h(x_j) = \sum_a \alpha_a N_a(x_j) @f$, a sparse mat-vec via
     *    the basis-evaluation matrix @f$ T_{ji} = N_i(x_j) @f$. A subclass
     *    may apply that projection, OR slice the real-vertex coefficients
     *    directly when the composite basis happens to interpolate at the
     *    mesh vertices. @ref LMEAssembler slices the ghost rows, and for
     *    the composite curved LME basis that slice is the EXACT projection,
     *    not an approximation: at a vertex @f$ x_j @f$ every Shepard-active
     *    chart projects @f$ x_j @f$ onto its own node-j position and the
     *    in-chart solve returns @f$ N_a(x_j) = \delta_{aj} @f$, so summed
     *    by the partition of unity the composite basis is interpolating
     *    (measured in @c test_lme_evaluate_modes_projection.cpp; see the
     *    R2 note on @ref LMEAssembler::evaluate_modes_at_vertices).
     *
     * @param V Vertex positions used for both assembly and target
     *          rendering (@f$ n_V \times 3 @f$). Must match the @p V
     *          passed to @ref assemble_K / @ref assemble_M.
     * @param F Mesh connectivity, same matrix as used for assembly.
     * @param mode_coefficients
     *          @f$ 3 n_V \times m @f$ matrix; column @c k is the
     *          eigenvector for mode @c k in the assembled-basis layout
     *          (row @c 3a+d is the coefficient on vertex @c a's
     *          displacement component @c d).
     * @return  @f$ 3 n_V \times m @f$ matrix in vertex-displacement
     *          coordinates; row @c 3a+d is the value of displacement
     *          component @c d at vertex @c a for mode @c k.
     */
    [[nodiscard]] virtual Eigen::MatrixXd evaluate_modes_at_vertices(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& mode_coefficients) const
    {
        (void)V;
        (void)F;
        return mode_coefficients;
    }
};

/**
 * @brief Loop-subdivision shell assembler (Cirak-Ortiz-Schroder 2000).
 *
 * Concrete @ref ShellAssembler wrapping the Loop-basis FEM exposed by
 * @ref chladni::shell::loop::assemble_stiffness_loop /
 * @ref chladni::shell::loop::assemble_mass_loop. The shipped defaults
 * are Stam exact evaluation on, @ref QuadratureRule::SevenPointDunavant
 * for both K and M, and @ref MassLumping::None (consistent Galerkin
 * mass — flipped from RowSum on 2026-05-17 late after the lumping
 * bias was diagnosed; see the @c m_lump field docstring below).
 *
 * @section quadrature Quadrature support
 *
 * All @ref QuadratureRule values are supported for both @c k_quad and
 * @c m_quad. The selected rule is forwarded to the Loop element kernels
 * (@ref chladni::shell::loop::element_stiffness_regular /
 * @ref chladni::shell::loop::element_mass_regular and the Stam
 * irregular-patch path), which evaluate
 * @ref chladni::shell::quadrature_points for whichever rule is chosen.
 * The shipped default is @ref QuadratureRule::SevenPointDunavant
 * (Dunavant degree-5, comfortably above the quartic Loop box-spline
 * basis); the lower-order 1-pt centroid / 3-pt edge-mid rules and the
 * 12-pt degree-6 rule are available for A/B and sensitivity studies.
 */
class LoopAssembler : public ShellAssembler {
public:
    /**
     * @brief Per-instance configuration of the Loop assembler.
     *
     * Defaults reproduce the shipped behavior bit-for-bit: changing a
     * single field is the intended way to A/B a formulation knob.
     */
    struct Params {
        /// Reference-triangle quadrature for K element integrals.
        /// Any @ref QuadratureRule is honored; default
        /// @ref QuadratureRule::SevenPointDunavant.
        QuadratureRule k_quad = QuadratureRule::SevenPointDunavant;

        /// Reference-triangle quadrature for M element integrals.
        /// Any @ref QuadratureRule is honored; default
        /// @ref QuadratureRule::SevenPointDunavant.
        QuadratureRule m_quad = QuadratureRule::SevenPointDunavant;

        /// Mass-matrix lumping applied to the consistent assembly.
        /// Default @ref MassLumping::None (consistent Galerkin) — the
        /// physically correct Galerkin formulation. RowSum was the
        /// shipped default 2026-05-13 → 2026-05-17 as a hot fix for
        /// high-@f$m@f$ doublet bumps, but lumped mass is well-known
        /// to lower eigenfrequencies broadly (lumped M majorises
        /// consistent M on smooth eigenvectors, so ω² = K/M drops).
        /// The most dramatic case measured so far is icosphere k=2
        /// at -35 % vs Wilkinson, but the user reported audible
        /// frequency lowering across mesh types (cylinder, plate),
        /// matching the textbook expectation. Flipped back to
        /// consistent on 2026-05-17 (late); see the @c [.experiment]
        /// mass-lumping probe in @c test_loop_sphere_projected.cpp
        /// for hard numbers on the icosphere case.
        MassLumping m_lump = MassLumping::None;

        /// Loop subdivision passes in the irregular branch (must be
        /// @f$ \ge 1 @f$). Ignored when the mesh has no extraordinary
        /// interior vertices (fast path).
        int n_passes = 1;

        /// Use Stam 1999 exact-evaluation for residual irregular
        /// sub-triangles in the subdivision branch. Default true.
        bool use_stam = true;
    };

    /// Construct from an explicit @ref Params. Validates inline
    /// (`n_passes >= 1`). Throws @c std::invalid_argument otherwise.
    explicit LoopAssembler(Params params);

    /// Default-constructed: equivalent to `LoopAssembler{Params{}}`.
    LoopAssembler() : LoopAssembler(Params{}) {}

    /// Read-only access to the configuration.
    [[nodiscard]] const Params& params() const noexcept { return params_; }

    [[nodiscard]] Eigen::SparseMatrix<double> assemble_K(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const ShellMaterial&   material) const override;

    [[nodiscard]] Eigen::SparseMatrix<double> assemble_M(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        double                 surface_density) const override;

    [[nodiscard]] std::string label() const override;

private:
    Params params_;
};

}  // namespace chladni::shell
