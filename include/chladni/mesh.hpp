#pragma once

/**
 * @file mesh.hpp
 * @brief Triangle mesh container and OBJ loader for the chladni pipeline.
 *
 * The triangle mesh is the universal input to the FEM stages:
 * @ref chladni::shell uses it as a thin shell. To minimise impedance
 * mismatches with Eigen-based
 * linear algebra and Spectra's eigensolver, vertices and faces are stored
 * as `Eigen::MatrixXd` (Nx3 double-precision) and `Eigen::MatrixXi` (Mx3
 * 32-bit signed indices) respectively, matching the convention used by
 * libigl and most of the geometry-processing literature.
 *
 * @section subset Supported OBJ subset
 * - vertex positions (`v x y z`)
 * - faces with three or more vertices (`f i j k …`); polygons with
 *   N > 3 vertices are fan-triangulated as (i_0, i_1, i_2),
 *   (i_0, i_2, i_3), …, (i_0, i_{N-2}, i_{N-1}).
 * - 1-based vertex indices (the OBJ spec); silently translated to 0-based.
 * - texture coordinates (`vt`), normals (`vn`), groups (`g`), and
 *   materials (`mtllib`, `usemtl`) are ignored.
 * - faces with negative (relative) indices are accepted.
 *
 * Any other directive is silently ignored (tinyobjloader skips unknown
 * keywords; its non-fatal warnings are not surfaced). A std::runtime_error
 * is raised only on a hard parse error, a file with no vertices, or a
 * non-triangle face surviving triangulation.
 */

#include <Eigen/Core>

#include <filesystem>

namespace chladni::mesh {

/**
 * @brief In-memory triangle mesh.
 *
 * The mesh is held by value and copies cheaply for small meshes; for
 * large meshes prefer pass-by-const-reference.
 */
struct TriMesh {
    /// Vertex positions, one per row: @f$ V \in \mathbb{R}^{n \times 3} @f$.
    Eigen::MatrixXd V;
    /// Triangle vertex indices, one per row: @f$ F \in \mathbb{Z}^{m \times 3} @f$.
    /// All indices are 0-based and satisfy @f$ 0 \le F_{ij} < n @f$.
    Eigen::MatrixXi F;

    /// Number of vertices.
    [[nodiscard]] Eigen::Index num_vertices() const noexcept { return V.rows(); }
    /// Number of triangles.
    [[nodiscard]] Eigen::Index num_faces()    const noexcept { return F.rows(); }
};

/**
 * @brief Quad-splitting scheme for the structured procedural
 *        generators (@ref generate_circular_disk, @ref generate_annulus,
 *        @ref generate_cylinder).
 *
 * The structured generators tile annular/axial strips with quads and
 * triangulate each quad. HOW the quads are split determines the
 * discrete symmetry group of the mesh — and through it the quality of
 * the degenerate Chladni doublets (2026-06-03 chirality
 * investigation; all numbers below are the 32x8 polar-disk Leissa n=2
 * fixture on the 2nd-order SME path):
 *
 * - @c Consistent — the legacy split: every quad cut by the same
 *   diagonal. Keeps the full C_N rotational symmetry (doublet
 *   frequencies degenerate to machine precision) but breaks EVERY
 *   reflection (face-set mirror-residual 0.86–1.0): the mesh has a
 *   handedness, the doublets have no mirror axis to pin them, and the
 *   displayed nodal patterns come out visibly skewed/chiral.
 *   rel_err +0.713 %. Uniform rim valence 4 — the only scheme the
 *   LOOP assembler's boundary augmentation accepts.
 *
 * - @c Checkerboard — alternate the diagonal per quad by
 *   (ring + j) parity. Restores an EXACT reflection (mirror-residual
 *   0) at unchanged vertex/triangle counts, but rim vertices then
 *   alternate boundary valence 3/5: a period-2 azimuthal modulation
 *   of the free edge that imprints visible high-frequency artefacts
 *   on the mode shapes near the rim. Frequencies on SME barely move
 *   (+0.705 %). Requires an even azimuthal count (an odd ring leaves
 *   a seam defect that silently restores the chirality — rejected
 *   loudly). Kept for symmetric-mesh experiments at fixed DOF; prefer
 *   @c UnionJack for actual Chladni work.
 *
 * - @c UnionJack — one centre vertex per quad, four triangles
 *   (criss-cross). Preserves the FULL dihedral symmetry (every quad
 *   identical, mirror-residual 0, no alternation anywhere; rim
 *   vertices uniformly boundary-valence 5) and is dramatically more
 *   accurate: rel_err +0.058 % — 12× better than @c Consistent at the
 *   same resolution arguments, and still 7.7× better than a
 *   @c Consistent mesh refined to the SAME vertex count (40x12,
 *   481 V, +0.450 %), so the gain is triangulation quality, not
 *   refinement. Works with any azimuthal count ≥ 3. Costs ~2× the
 *   triangles of @c Consistent at equal resolution arguments.
 *
 * CAVEAT (both symmetric schemes): boundary vertices are no longer
 * uniformly valence 4, which @c augment_for_loop_boundary rejects
 * (the LOOP assembler throws). The 1st-order LME ghost path handles
 * all three splits cleanly since 2026-06-03 (sub-1 % on the 32x8
 * Leissa disk; the earlier −19.5 % / −31.7 % failures were the
 * uniform-β closed-form-derivative deviation, fixed faithfully per
 * Millán 2011 App A — see the [quadsplit] regression gates in
 * tests/shell/test_lme_free_edge_circular_plate.cpp).
 */
enum class QuadSplit {
    Consistent,    ///< Legacy single-diagonal split (chiral).
    Checkerboard,  ///< Parity-alternating diagonals (exact mirror).
    UnionJack,     ///< Centre vertex + 4 triangles per quad (full D_N).
};

/**
 * @brief Generate a procedural open cylindrical tube as a triangle mesh.
 *
 * Produces an open tube of radius @p radius and axial length @p length
 * (no end caps). The mid-surface is sampled with @p n_around vertices
 * per ring and @p n_along + 1 rings (i.e. @p n_along axial segments)
 * laid out at uniform z spacing in @f$[0, L]@f$. Vertex layout is
 * row-major in (axial ring, around-the-ring index): vertex
 * @f$(j \cdot n_\text{around} + i)@f$ has position
 * @f[
 *   \bigl(R\,\cos(2\pi i / n_\text{around}),\;
 *         R\,\sin(2\pi i / n_\text{around}),\;
 *         L\,j / n_\text{along}\bigr).
 * @f]
 * By default each axial-circumferential quad is split along the
 * consistent diagonal @f$R_j[i] \to R_{j+1}[i+1]@f$, matching the
 * topology of the bundled `models/cylinder.obj`. The result is a
 * closed (manifold) tube apart from the two open boundary loops at
 * @f$z = 0@f$ and @f$z = L@f$.
 *
 * Total vertex count is @f$ n_\text{around} \cdot (n_\text{along} + 1) @f$
 * and triangle count is @f$ 2 \cdot n_\text{around} \cdot n_\text{along} @f$.
 *
 * Use this to feed `chladni::shell::compute_shell_modes` clean
 * parametric input for FEM-vs-analytic comparison; see
 * @ref chladni::analytical::free_free_cylindrical_shell_inextensional_angular_frequencies.
 *
 * With @ref QuadSplit::UnionJack the counts grow to
 * @f$ n_\text{around} \cdot (2 n_\text{along} + 1) @f$ vertices and
 * @f$ 4 \cdot n_\text{around} \cdot n_\text{along} @f$ triangles (one
 * centre vertex + 4 triangles per quad); the boundary loops are
 * unchanged.
 *
 * @param radius     Mid-surface radius @f$R@f$ (m), @f$> 0@f$.
 * @param length     Axial length @f$L@f$ (m), @f$> 0@f$.
 * @param n_around   Number of vertices per ring; must be @f$\ge 3@f$
 *                   (and EVEN for @ref QuadSplit::Checkerboard).
 * @param n_along    Number of axial segments (rings = @p n_along + 1);
 *                   must be @f$\ge 1@f$.
 * @param split      Quad-splitting scheme — see @ref QuadSplit for the
 *                   symmetry/chirality/assembler-compatibility
 *                   trade-offs.
 *
 * @return TriMesh with the described topology.
 *
 * @throws std::invalid_argument
 *         on non-positive @p radius / @p length, on resolution
 *         arguments below the documented minima, or on
 *         @ref QuadSplit::Checkerboard with odd @p n_around.
 */
TriMesh generate_cylinder(double radius,
                          double length,
                          int n_around,
                          int n_along,
                          QuadSplit split = QuadSplit::Consistent);

/**
 * @brief Generate a flat hex-triangulated rectangular plate suitable
 *        for the Loop subdivision FEM pipeline.
 *
 * The plate occupies @f$[0, a] \times [0, b]@f$ in the xy-plane at
 * @f$z = 0@f$. The interior is a uniform-spacing rectangular grid of
 * @f$(n_x + 1) \times (n_y + 1)@f$ candidate vertex positions, each
 * cell triangulated along the up-right diagonal. Two of the four
 * candidate corner vertices, @f$(n_x, 0)@f$ and @f$(0, n_y)@f$, are
 * **removed** along with the unique single-triangle face that contains
 * each — chamfering the plate into a hexagonal-rectangle shape so that
 * the remaining boundary corners all have valence 3 (instead of the
 * mixed 2/3 the unmodified pattern would produce). The chamfering
 * removes 2 vertices and 2 triangles total.
 *
 * Topology after chamfering, for any @f$n_x, n_y \ge 2@f$:
 *  - all interior vertices (those at @f$1 \le i \le n_x-1@f$ and
 *    @f$1 \le j \le n_y-1@f$) have valence 6,
 *  - mid-boundary vertices (on a rectangle edge but not a corner of
 *    the chamfered hexagonal-rectangle) have valence 4,
 *  - the 6 corners of the chamfered shape have valence 3 — the 2
 *    original valence-3 rectangle corners plus 4 new valence-3
 *    corners adjacent to the removed vertices.
 *
 * That valence pattern is exactly what the L.5c.1 augmentation expects
 * (it accepts valence-3 and valence-4 boundary vertices and rejects
 * valence-2). The shape difference between this hex-rectangle and a
 * true rectangle of dimensions @f$a \times b@f$ is small for moderate
 * @f$n_x, n_y@f$: 2 missing corner vertices reduce the plate area by
 * @f$(a / n_x)(b / n_y) / 2 \cdot 2@f$ — a triangle of area
 * @f$ab / (n_x n_y)@f$, vanishing as @f$O(1 / (n_x n_y))@f$.
 *
 * Vertex layout (row-major in (j, i) skipping the 2 removed corners):
 *  - row 0 of the grid (j = 0): vertices (0, 0), (1, 0), …, (n_x-1, 0).
 *    Vertex (n_x, 0) is omitted.
 *  - rows 1 .. n_y-1: full row of n_x+1 vertices each.
 *  - row n_y: vertex (0, n_y) is omitted; the rest are (1, n_y), …,
 *    (n_x, n_y).
 *
 * Total vertex count: @f$(n_x + 1)(n_y + 1) - 2@f$.
 * Total triangle count: @f$2 n_x n_y - 2@f$ (full grid 4 sub-triangles
 * per cell minus the 2 removed by chamfering).
 *
 * Use this to feed `chladni::shell::compute_shell_modes_loop` when
 * comparing against thin-plate analytic references.
 *
 * @param length_a   Plate extent along the x-axis (m), @f$> 0@f$.
 * @param length_b   Plate extent along the y-axis (m), @f$> 0@f$.
 * @param n_x        Number of cells along the x-axis; must be @f$\ge 2@f$.
 *                   (n_x = 1 would put two chamfered corners on the
 *                   same edge and produces a degenerate mesh.)
 * @param n_y        Number of cells along the y-axis; must be @f$\ge 2@f$.
 *
 * @return TriMesh with the described chamfered-rectangle topology,
 *         vertices in the xy-plane, all faces CCW seen from +z.
 *
 * @throws std::invalid_argument
 *         on non-positive @p length_a / @p length_b, or on resolution
 *         arguments below the documented minima.
 */
TriMesh generate_flat_plate(double length_a,
                            double length_b,
                            int    n_x,
                            int    n_y);

/**
 * @brief Generate a closed icosphere mesh of radius @p radius.
 *
 * Starts from a regular icosahedron (12 vertices, 20 faces) and applies
 * @p n_subdivisions rounds of edge-midpoint subdivision, projecting
 * each new vertex onto the sphere of radius @p radius after each round.
 * After @f$k@f$ subdivisions the mesh has
 *   @f$ V_k = 12 + 30 \frac{4^k - 1}{3} @f$ vertices, and
 *   @f$ F_k = 20 \cdot 4^k @f$ faces.
 *
 * Concretely: 12V/20F at k=0, 42V/80F at k=1, 162V/320F at k=2, 642V/
 * 1280F at k=3, 2562V/5120F at k=4.
 *
 * Topology after any positive @p n_subdivisions:
 *  - the 12 vertices of the original icosahedron remain at valence 5
 *    (extraordinary in the Loop / Cirak-Ortiz sense),
 *  - every other vertex (one per edge of the previous level, projected
 *    onto the sphere) has valence 6,
 *  - the mesh is closed (no boundary edges) and CCW-wound (outward
 *    face normals point away from the origin).
 *
 * Each parent face is split into 4 sub-faces (3 corner sub-triangles
 * around each parent vertex + 1 central medial sub-triangle), matching
 * the Loop subdivision combinatorics. The vertex POSITIONS are placed
 * by spherical projection (geometric, not Loop's smoothing masks) so
 * the mesh remains exactly on the sphere.
 *
 * Pipeline considerations for @c assemble_stiffness_loop:
 *  - At @c n_subdivisions = 0 (bare icosahedron) every triangle has 3
 *    extraordinary corners — VIOLATES the Cirak-Ortiz §4.6 step 1
 *    prerequisite that each triangle has at most one extraordinary
 *    corner. The L.3.4 path still produces *some* @c K but drops the
 *    majority of the surface energy.
 *  - At @c n_subdivisions @f$\ge 1@f$ each parent face has been split,
 *    so the corner sub-faces touch at most one valence-5 vertex — the
 *    prerequisite is satisfied and one or two passes of subdivision
 *    inside @c assemble_stiffness_loop give clean accuracy.
 *
 * @param radius          Sphere radius (m), @f$> 0@f$.
 * @param n_subdivisions  Number of edge-midpoint subdivision rounds;
 *                        must be @f$\ge 0@f$.
 *
 * @return TriMesh with all vertices on the sphere of radius @p radius.
 *
 * @throws std::invalid_argument
 *         on non-positive @p radius or negative @p n_subdivisions.
 */
TriMesh generate_icosphere(double radius, int n_subdivisions);

/**
 * @brief Generate a flat circular disk in the z=0 plane.
 *
 * Polar-grid triangulation. One central vertex at the origin, plus
 * @p n_radial concentric rings of @p n_azimuthal vertices each evenly
 * spaced in angle from 0 to @f$2\pi@f$ (exclusive). Radii are uniform
 * in @f$[0, R]@f$.
 *
 * Topology after construction, for @f$n_\text{azimuthal} \ge 3@f$ and
 * @f$n_\text{radial} \ge 1@f$:
 *  - 1 central vertex at @f$(0, 0, 0)@f$ with valence
 *    @f$n_\text{azimuthal}@f$ — **extraordinary in the Loop /
 *    Cirak-Ortiz sense** (handled by the L.3.4 / Stam path).
 *  - @f$(n_\text{radial} - 1) \cdot n_\text{azimuthal}@f$ interior
 *    ring vertices with valence 6 (regular).
 *  - @f$n_\text{azimuthal}@f$ outer-ring vertices with boundary
 *    valence 4 (2 boundary edges + 2 interior edges) — supported by
 *    @c augment_for_loop_boundary.
 *  - mesh is open (one boundary loop on the outer ring).
 *  - all faces CCW seen from +z.
 *
 * Total vertex count: @f$ 1 + n_\text{radial} \cdot n_\text{azimuthal} @f$.
 * Total triangle count:
 *   @f$ n_\text{azimuthal} (2 n_\text{radial} - 1) @f$
 *   (@c n_azimuthal triangles in the central fan +
 *    @c 2*n_azimuthal per annular strip between adjacent rings).
 *
 * Use this to feed @c chladni::shell::compute_shell_modes_loop when
 * comparing against Leissa's free-edge circular plate analytic
 * (the canonical Chladni-figure fixture).
 *
 * With @ref QuadSplit::UnionJack the counts grow to
 * @f$ 1 + (2 n_\text{radial} - 1) \cdot n_\text{azimuthal} @f$
 * vertices and
 * @f$ n_\text{azimuthal} (4 n_\text{radial} - 3) @f$ triangles (one
 * centre vertex + 4 triangles per strip quad; the central fan is
 * unchanged — it is mirror-invariant as-is). The rim stays a single
 * boundary loop of @p n_azimuthal edges.
 *
 * @param radius        Outer radius @f$R@f$ (m), @f$> 0@f$.
 * @param n_azimuthal   Number of vertices per ring, @f$\ge 3@f$
 *                      (and EVEN for @ref QuadSplit::Checkerboard).
 * @param n_radial      Number of rings, @f$\ge 1@f$.
 * @param split         Quad-splitting scheme — see @ref QuadSplit for
 *                      the symmetry/chirality/assembler-compatibility
 *                      trade-offs (measured on THIS fixture).
 *
 * @return TriMesh with vertices in the z=0 plane, CCW faces.
 *
 * @throws std::invalid_argument
 *         on non-positive @p radius, resolution below the minima, or
 *         @ref QuadSplit::Checkerboard with odd @p n_azimuthal.
 */
TriMesh generate_circular_disk(double radius,
                               int    n_azimuthal,
                               int    n_radial,
                               QuadSplit split = QuadSplit::Consistent);

/**
 * @brief Generate a flat regular-n-gon disk in the z=0 plane with an
 *        isotropic, near-equilateral interior triangulation.
 *
 * The rim is the regular @p n_boundary-gon inscribed in @f$r = R@f$ —
 * its @p n_boundary vertices are kept exactly on the circle without
 * snapping or rescaling. The polygon edge length
 * @f$s = 2R\sin(\pi/n_\text{boundary})@f$ sets the target interior
 * edge length. Concentric inner rings are placed at
 * @f$r_m = R - m\,s\,\sqrt{3}/2@f$ for @f$m = 1, 2, \dots@f$ (the
 * equilateral-lattice row spacing) until the next radius falls below a
 * half-step from the origin; the centre is then closed with a single
 * vertex. Each inner ring @f$m@f$ carries
 * @f$N_m = \max(3, \text{round}(2\pi r_m / s))@f$ vertices uniformly
 * spaced, keeping the azimuthal step within a few percent of @f$s@f$
 * at every depth. Strips between rings with different vertex counts
 * are stitched with the standard monotone-advance pattern (alternating
 * "outer-apex" and "inner-apex" triangles), so every interior vertex
 * has valence near 6 and every interior triangle is near-equilateral —
 * none of the sliver / valence-@p n_boundary central hub artefacts of
 * the polar @ref generate_circular_disk.
 *
 * Topology guarantees:
 *  - @p n_boundary rim vertices on @f$r = R@f$, ordered CCW from
 *    angle 0; these are vertex indices @f$0 \dots n_\text{boundary}-1@f$.
 *  - one boundary loop (the rim), mesh is open.
 *  - all faces CCW seen from +z.
 *
 * @param radius      Circumradius @f$R@f$ (m), @f$> 0@f$.
 * @param n_boundary  Number of rim vertices, @f$\ge 3@f$.
 *
 * @return TriMesh with vertices in the z=0 plane, CCW faces.
 *
 * @throws std::invalid_argument
 *         on non-positive @p radius or @p n_boundary @f$< 3@f$.
 */
TriMesh generate_disk_iso(double radius, int n_boundary);

/**
 * @brief Generate a flat regular-hexagonal disk in the z=0 plane,
 *        triangulated as a clipped hex lattice.
 *
 * Every lattice vertex on hex-layer @f$m \le L@f$ is included
 * (@f$L = n_\text{layers}@f$), and the standard "up + down" triangle
 * pair is emitted for every (a, b) whose 3 neighbours are also in the
 * disk. Lattice spacing is @f$s = R/L@f$. After construction, the
 * outermost ring (layer L) is radially snapped onto @f$r = R@f$ so the
 * silhouette is a circle, not a hexagon — corners were already on the
 * circle and don't move; edge midpoints move outward from
 * @f$R \sqrt{3}/2@f$ to @f$R@f$. Interior topology is untouched: every
 * non-rim vertex stays a clean hex-lattice point (valence 6 on the
 * Loop subdivision graph), and only the outermost ring of triangles
 * gets locally distorted by the radial snap (no slivers, no inversions
 * at any reasonable L).
 *
 * Topology — this is the cleanest possible Loop subdivision input
 * for the disk. The 6 hexagon corners (the only valence-3 boundary
 * vertices) are CHAMFERED away — dropped from the vertex set and bridged
 * by a single triangle across the chord between their two rim neighbours
 * — leaving a clean valence-4 boundary at the price of 6 interior
 * valence-5 vertices (which Loop / Stam handle cleanly):
 *  - centre vertex (and every interior lattice vertex) is valence 6,
 *    except the 6 layer-(L-1) corner vertices, which are valence 5.
 *  - the rim is a single boundary loop of @f$6(L-1)@f$ edges, all
 *    boundary valence 4.
 *
 * Counts (after dropping the 6 corners; @f$L \ge 2@f$):
 *  - vertices: @f$1 + 3L(L+1) - 6 = 3L^2 + 3L - 5@f$.
 *  - triangles: @f$6L^2 - 6 = 6(L^2 - 1)@f$.
 *  - boundary edges: @f$6(L-1)@f$.
 *
 * Use this as a diagnostic against @ref generate_disk_iso (distributed
 * valence-5/7 sprinkled through the interior) or
 * @ref generate_circular_disk (one valence-@f$n_\text{az}@f$ hub at the
 * centre). If mode shapes are clean on the hex disk and dirty on either
 * of the others, the issue is intrinsic to the irregular-vertex
 * distribution of that mesh, not to the FEM pipeline.
 *
 * @param radius    Circumradius @f$R@f$ (m), @f$> 0@f$ — distance from
 *                  origin to the 6 hexagon corners.
 * @param n_layers  Number of hexagonal subdivision layers, @f$\ge 2@f$.
 *                  @f$L = 2@f$ is the smallest non-degenerate disk
 *                  (13 vertices, 18 triangles); larger L refines.
 *                  @f$L = 1@f$ is rejected — all six layer-1 points are
 *                  hex corners, so chamfering would leave a single
 *                  degenerate vertex.
 *
 * @return TriMesh with vertices in the z=0 plane, CCW faces.
 *
 * @throws std::invalid_argument on non-positive @p radius or
 *         @p n_layers @f$< 2@f$.
 */
TriMesh generate_disk_hex(double radius, int n_layers);

/**
 * @brief Generate a flat annular (ring) plate in the z=0 plane.
 *
 * Polar-grid triangulation between inner radius @p radius_inner and
 * outer radius @p radius_outer. @p n_radial concentric rings of
 * @p n_azimuthal vertices each, evenly spaced in angle from 0 to
 * @f$2\pi@f$ (exclusive). Radii are uniform in
 * @f$[r_\text{in}, r_\text{out}]@f$ — ring 0 sits at @c radius_inner,
 * ring @c n_radial-1 at @c radius_outer.
 *
 * Topology after construction, for @f$n_\text{azimuthal} \ge 3@f$ and
 * @f$n_\text{radial} \ge 2@f$:
 *  - inner boundary ring: @c n_azimuthal vertices with boundary valence 4.
 *  - interior ring vertices: valence 6 (regular).
 *  - outer boundary ring: @c n_azimuthal vertices with boundary valence 4.
 *  - mesh is doubly-bounded (one boundary loop on each ring).
 *  - all faces CCW seen from +z.
 *
 * Total vertex count: @f$ n_\text{radial} \cdot n_\text{azimuthal} @f$.
 * Total triangle count: @f$ 2 \cdot n_\text{azimuthal} \cdot (n_\text{radial} - 1) @f$.
 *
 * Use this to feed @c chladni::shell::compute_shell_modes_loop when
 * comparing against Leissa's clamped-clamped annular plate analytic.
 *
 * With @ref QuadSplit::UnionJack the counts grow to
 * @f$ (2 n_\text{radial} - 1) \cdot n_\text{azimuthal} @f$ vertices
 * and @f$ 4 n_\text{azimuthal} (n_\text{radial} - 1) @f$ triangles
 * (one centre vertex + 4 triangles per quad); both rims stay
 * boundary loops of @p n_azimuthal edges each.
 *
 * @param radius_outer  Outer radius @f$a@f$ (m), @f$> 0@f$.
 * @param radius_inner  Inner radius @f$b@f$ (m), @f$0 < b < a@f$.
 * @param n_azimuthal   Number of vertices per ring, @f$\ge 3@f$
 *                      (and EVEN for @ref QuadSplit::Checkerboard).
 * @param n_radial      Number of rings, @f$\ge 2@f$.
 * @param split         Quad-splitting scheme — see @ref QuadSplit for
 *                      the symmetry/chirality/assembler-compatibility
 *                      trade-offs.
 *
 * @return TriMesh with vertices in the z=0 plane, CCW faces.
 *
 * @throws std::invalid_argument
 *         on non-positive @p radius_outer, @p radius_inner outside
 *         @f$(0, a)@f$, resolution below the minima, or
 *         @ref QuadSplit::Checkerboard with odd @p n_azimuthal.
 */
TriMesh generate_annulus(double radius_outer,
                         double radius_inner,
                         int    n_azimuthal,
                         int    n_radial,
                         QuadSplit split = QuadSplit::Consistent);

/**
 * @brief Load a triangle mesh from a Wavefront `.obj` file.
 *
 * @param path Filesystem path to a readable OBJ file.
 *
 * @return TriMesh whose `V` and `F` are populated from the file. Faces
 *         are validated post-load: every index must be in @f$ [0, n) @f$.
 *
 * @throws std::runtime_error
 *         on any of: file not found / not readable; a tinyobjloader
 *         parse error (malformed file); out-of-range face index;
 *         degenerate face (two of three indices are equal); empty mesh
 *         (no vertices). Unknown/unsupported directives are NOT an error
 *         — tinyobjloader silently skips them (see @ref subset above).
 *
 * @see TriMesh, supported OBJ subset under @ref subset.
 */
TriMesh load_obj(const std::filesystem::path& path);

}  // namespace chladni::mesh
