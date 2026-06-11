/**
 * @file test_static_obstacle_course.cpp
 * @brief Paper-faithful STATIC shell benchmarks (Belytschko obstacle course).
 *
 * Motivation. The free-free cylinder modal test
 * (@c test_modes_vs_free_free_cylinder_analytic.cpp) shows the curved-shell
 * LME assembler MEMBRANE-LOCKS on inextensional ovalling modes (see the
 * @c [lme_scale] / @c [lme_prescribed_mem] diagnostics and the
 * @c chladni-lme-cylinder-locking memory). But that is a *free-free modal*
 * problem — a regime Millán-Rosolen-Arroyo 2011 never validates. Every
 * numerical example in §4 of the paper is a STATIC equilibrium problem
 * @f$ K U = f @f$ with Dirichlet supports (diaphragms or clamped), and
 * "convergence" there means the relative error of a single displacement
 * under a prescribed load vs an overkill/reference value.
 *
 * To decide whether we have a genuine reproduction *bug* in the membrane
 * operator versus an *out-of-regime usage* (free-free modal), this file
 * reproduces the paper's own yardstick: the classic pinched cylinder with
 * rigid end diaphragms — the canonical membrane-locking benchmark, part of
 * the obstacle course the paper claims to pass (Belytschko et al. [27],
 * MacNeal-Harder 1985).
 *
 *   geometry/material (MacNeal-Harder):
 *     L = 600, R = 300, t = 3, E = 3e6, ν = 0.3
 *     two opposite radial point loads P = 1 at mid-length (θ = 0, π)
 *     rigid diaphragm at both ends: u_x = u_y = 0 (u_z free)
 *     reference radial deflection under the load: w_ref = 1.8248e-5
 *
 * RESOLVED (2026-05-29): the bending under-stiffness below was FIXED by
 * switching the curved LME path to the paper's value-based neighbour cutoff
 * (Params::tol_lme, Millán Eq. 2) instead of the fixed r_cut_mult_curved=1.4
 * that starved the bending Hessian. Scordelis-Lo now matches the paper
 * (~1.02×) and structured-plate bending is ~0.97×. The findings below are
 * the diagnosis that led to that fix (kept as the rationale); references to
 * "r_cut_mult_curved=1.4 / too soft" describe the PRE-FIX state.
 *
 * WHAT THIS SESSION FOUND (2026-05-29). Building the benchmark surfaced a
 * more fundamental defect than the membrane locking it set out to probe:
 *
 *   1. The HARNESS is correct. Loop reproduces the exact Kirchhoff
 *      simply-supported-plate coefficients (point + uniform load) and the
 *      prescribed-bending-field energy to <1%.
 *
 *   2. The FLAT LME path (@c use_curved_shell=false) is also correct
 *      (<1.5% on plate deflection and bending energy).
 *
 *   3. The CURVED-shell LME path (@c use_curved_shell=true — the shipped
 *      default used for ALL curved/cylinder work) is ~36-50% TOO SOFT in
 *      BENDING *even on a flat plate*, where it must reduce exactly to the
 *      flat-plate operator. Measured four independent ways, all consistent:
 *        - SS-plate central deflection:        1.56× analytic (too soft)
 *        - SS-plate uniform-load compliance:   1.57× Loop (basis-invariant)
 *        - prescribed sin·sin bending energy:  0.639× analytic
 *        - existing [lme_bend_flat] strip:     0.51-0.53× analytic
 *      This is INDEPENDENT of and IN ADDITION TO the documented membrane
 *      locking. It was already visible in the [lme_bend_flat] control but
 *      never recognised as a core defect.
 *
 *   ROOT CAUSE (confirmed in code + by r_cut sweep). It is NOT a PoU
 *   product-rule omission: the curved assembly is a correct
 *   PoU-split-of-the-integral (Σ_A ∫ w_A B_A^T C B_A), each patch using
 *   its own single-chart Hessian — no composite to differentiate. The
 *   real difference from the (correct) flat path is the BASIS STENCIL:
 *     - flat path evaluates the LME Hessian against ALL nodes at
 *       r_cut_mult = 4.0 (wide stencil → accurate 2nd derivative);
 *     - curved path evaluates against a per-patch k-ring set at
 *       r_cut_mult_curved = 1.4 (tiny stencil → under-resolved Hessian).
 *   C_bend is identical between paths. The bending-energy sweep over
 *   r_cut_mult_curved confirms it directly (ncell=24, prescribed sin·sin):
 *       1.4 → 0.639×   2.0 → 0.928×   3.0 → 0.966×   4.0 → 0.966×
 *   So widening the stencil to ~3-4 restores correct bending.
 *
 *   The catch: r_cut_mult_curved = 1.4 was deliberately tuned to MINIMISE
 *   membrane locking on the free-free cylinder (wider r_cut worsens the
 *   spurious membrane — see chladni-lme-cylinder-locking). So bending
 *   accuracy and membrane-locking mitigation are in DIRECT TENSION through
 *   this one knob. The paper-faithful setting is a wide stencil (its β=0.8
 *   with adequate patch support gives correct bending); the residual
 *   inextensional-membrane locking is then the genuine 1st-order-LME
 *   limitation the paper acknowledges (§4.1.2) — the proper target for a
 *   selective reduced-integration membrane fix.
 *
 * The decisive evidence is therefore the flat-plate diagnostics
 * (@c [ss_plate], @c [ss_plate_bend_energy], @c [ss_plate_config]) — NOT
 * the pinched cylinder, whose point-load + non-interpolating read make it
 * a poor discriminator (see its annotation).
 *
 *   4. PAPER-MATCH CONFIRMED on the paper's own cylindrical benchmark.
 *      The Scordelis-Lo roof (§4.3, @c [scordelis_lo]) reproduces the
 *      reference free-edge deflection 0.3024 (paper overkill 0.300575):
 *        Loop                  → 0.984× ref (converged)
 *        LME wide (rcut=4.0)   → 1.022× ref (converged from above)
 *        LME default (rcut=1.4)→ 1.225× ref (~22% too soft, bending starve)
 *      So the curved-shell LME operator IS faithful to the paper once the
 *      stencil is wide enough to resolve bending; Scordelis-Lo is
 *      membrane+bending dominated (not inextensional) so the membrane
 *      locking does not bite — exactly as the paper notes. The remaining
 *      free-free-cylinder failure is then PURELY the inextensional-membrane
 *      locking, the proper target for selective reduced membrane
 *      integration.
 *
 * @note LME is non-interpolating; the assembler treats real-vertex
 *       coefficients as nodal displacement amplitudes (weak Kronecker-delta
 *       — the same assumption its @c evaluate_modes_at_vertices slice
 *       relies on). For the flat plate this is exact at the boundary and
 *       the bending-energy probe sidesteps the read entirely; for the
 *       pinched cylinder it is one of several confounds.
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/loop.hpp>
#include <chladni/shell/lme.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// MacNeal-Harder pinched-cylinder material (E = 3e6, ν = 0.3). Density is
/// irrelevant for the static solve but must be positive for the calibrator.
chladni::IsotropicMaterial pinched_cylinder_material()
{
    return {.youngs_modulus = 3.0e6,
            .poisson_ratio  = 0.30,
            .density        = 1.0};
}

/**
 * @brief Build a cylindrical-roof arc mesh for the Scordelis-Lo benchmark.
 *
 * Cylinder axis along +x, length @p L (x ∈ [0, L]). Cross-section in the
 * y-z plane, radius @p R, vertical = +z (crown at the top). The arc spans
 * the angular half-range @p half_angle_deg about the crown:
 * φ ∈ [-half, +half] measured from +z. Vertex @f$(j\,(n_\text{arc}+1)+i)@f$
 * sits at @f$(L\,j/n_\text{axial},\; R\sin\varphi_i,\; R\cos\varphi_i)@f$.
 * Open arc (NOT wrapped): the i=0 and i=n_arc edges are the free straight
 * edges; the j=0 and j=n_axial rings are the diaphragm-supported curved
 * ends.
 *
 * @param n_arc    arc subdivisions (vertices per ring = n_arc + 1).
 * @param n_axial  axial subdivisions (rings = n_axial + 1).
 */
chladni::mesh::TriMesh build_cylindrical_roof(
    double R, double L, double half_angle_deg, int n_arc, int n_axial)
{
    chladni::mesh::TriMesh m;
    const int per_ring = n_arc + 1;
    const Eigen::Index n_v = static_cast<Eigen::Index>(per_ring) * (n_axial + 1);
    m.V.resize(n_v, 3);
    const double half = half_angle_deg * std::numbers::pi / 180.0;
    for (int j = 0; j <= n_axial; ++j) {
        const double x = L * static_cast<double>(j) / n_axial;
        for (int i = 0; i < per_ring; ++i) {
            const double phi = -half + 2.0 * half * static_cast<double>(i) / n_arc;
            const Eigen::Index v = static_cast<Eigen::Index>(j) * per_ring + i;
            m.V(v, 0) = x;
            m.V(v, 1) = R * std::sin(phi);
            m.V(v, 2) = R * std::cos(phi);
        }
    }
    m.F.resize(2 * n_arc * n_axial, 3);
    Eigen::Index t = 0;
    for (int j = 0; j < n_axial; ++j) {
        for (int i = 0; i < n_arc; ++i) {
            const Eigen::Index a = static_cast<Eigen::Index>(j) * per_ring + i;
            const Eigen::Index b = a + 1;
            const Eigen::Index c = a + per_ring;
            const Eigen::Index d = c + 1;
            // consistent diagonal a -> d (matches generate_cylinder)
            m.F(t, 0) = a; m.F(t, 1) = b; m.F(t, 2) = d; ++t;
            m.F(t, 0) = a; m.F(t, 1) = d; m.F(t, 2) = c; ++t;
        }
    }
    return m;
}

/// MacNeal-Harder pinched-hemisphere material (E = 6.825e7, ν = 0.3).
chladni::IsotropicMaterial pinched_hemisphere_material()
{
    return {.youngs_modulus = 6.825e7,
            .poisson_ratio  = 0.30,
            .density        = 1.0};
}

/**
 * @brief Quarter pinched-hemisphere mesh (MacNeal-Harder / paper §4.4).
 *
 * Hemisphere of radius @p R with an @p theta_hole_deg polar hole, restricted
 * to the φ ∈ [0°, 90°] quadrant for symmetry. Lat-long structured grid:
 * vertex (j*(n_phi+1)+i) at polar angle θ_j ∈ [hole, 90°] and azimuth
 * φ_i ∈ [0°, 90°], position R·(sinθ cosφ, sinθ sinφ, cosθ).
 *   - j=0 ring: hole edge (free).   j=n_theta ring: equator (free).
 *   - i=0 column: φ=0 symmetry plane (xz). i=n_phi column: φ=90 (yz).
 * The two diametral load points are the equator corners (θ=90): φ=0
 * (outward, +x radial) and φ=90 (inward, −y radial).
 */
chladni::mesh::TriMesh build_quarter_hemisphere(
    double R, double theta_hole_deg, int n_theta, int n_phi)
{
    chladni::mesh::TriMesh m;
    const int per_row = n_phi + 1;
    const Eigen::Index n_v =
        static_cast<Eigen::Index>(n_theta + 1) * per_row;
    m.V.resize(n_v, 3);
    const double th0 = theta_hole_deg * std::numbers::pi / 180.0;
    const double th1 = 0.5 * std::numbers::pi;            // 90°
    const double ph1 = 0.5 * std::numbers::pi;            // 90°
    for (int j = 0; j <= n_theta; ++j) {
        const double th = th0 + (th1 - th0) * j / n_theta;
        for (int i = 0; i < per_row; ++i) {
            const double ph = ph1 * i / n_phi;
            const Eigen::Index v =
                static_cast<Eigen::Index>(j) * per_row + i;
            m.V(v, 0) = R * std::sin(th) * std::cos(ph);
            m.V(v, 1) = R * std::sin(th) * std::sin(ph);
            m.V(v, 2) = R * std::cos(th);
        }
    }
    m.F.resize(2 * n_theta * n_phi, 3);
    Eigen::Index t = 0;
    for (int j = 0; j < n_theta; ++j) {
        for (int i = 0; i < n_phi; ++i) {
            const Eigen::Index a =
                static_cast<Eigen::Index>(j) * per_row + i;
            const Eigen::Index b = a + 1;
            const Eigen::Index c = a + per_row;
            const Eigen::Index d = c + 1;
            m.F(t, 0) = a; m.F(t, 1) = b; m.F(t, 2) = d; ++t;
            m.F(t, 0) = a; m.F(t, 1) = d; m.F(t, 2) = c; ++t;
        }
    }
    return m;
}

/**
 * @brief Quarter pinched-hemisphere on a GEODESIC point set — the paper's
 *        own discretization (Millán-Rosolen-Arroyo 2011 §2.6: "subdivisions
 *        of an octahedron following Loop's scheme, relocate the points on
 *        the unit sphere"), as opposed to the degenerate lat-long grid of
 *        @ref build_quarter_hemisphere.
 *
 * Subdivides the single spherical octant spanned by +x,+y,+z (one
 * octahedral face) into a quasi-uniform geodesic triangulation, then
 * removes the polar cap (θ < @p theta_hole_deg about +z) for the hole.
 * The octant's three edges lie EXACTLY on the coordinate planes, so the
 * model boundaries are clean: z=0 (equator), y=0 (φ=0 symmetry), x=0
 * (φ=90 symmetry), plus the hole rim. The two load corners (R,0,0) and
 * (0,R,0) are exact vertices. @p n is the number of subdivisions per
 * octant edge → (n+1)(n+2)/2 octant nodes before the hole is cut.
 *
 * @p conform_rim — CONFORMAL hole cut (2026-06-07, [hemi_rim_snap]):
 * after the face-dropped cut, project every hole-rim boundary vertex
 * onto the exact θ = @p theta_hole_deg circle (keeping φ). The default
 * face-dropped rim is a staircase polyline whose turn angles (47–70°)
 * spuriously corner-classify 4→12 rim nodes (d=0) under the SME §3.2.2
 * recipe ([hemi_corner_cls]); the paper's own discretizations have a
 * clean rim, so the conformal variant is the FIXTURE-faithful one.
 * Kept opt-in until the A/B is read — the shipped gates pin the
 * face-dropped mesh.
 */
chladni::mesh::TriMesh build_quarter_hemisphere_geodesic(
    double R, double theta_hole_deg, int n, bool conform_rim = false)
{
    const Eigen::Vector3d P0(R, 0, 0), P1(0, R, 0), P2(0, 0, R);
    // Barycentric lattice id(i,j), i+j<=n, k=n-i-j; offset[i] row start.
    std::vector<int> off(static_cast<std::size_t>(n + 2), 0);
    for (int i = 0; i <= n; ++i)
        off[static_cast<std::size_t>(i + 1)] = off[static_cast<std::size_t>(i)] + (n - i + 1);
    auto id = [&](int i, int j) { return off[static_cast<std::size_t>(i)] + j; };
    const int n_grid = off[static_cast<std::size_t>(n + 1)];

    Eigen::MatrixXd Vg(n_grid, 3);
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n - i; ++j) {
            const int k = n - i - j;
            Eigen::Vector3d p =
                (double(i) * P0 + double(j) * P1 + double(k) * P2);
            p = R * p.normalized();           // relocate onto the sphere
            Vg.row(id(i, j)) = p.transpose();
        }

    const double cos_hole = std::cos(theta_hole_deg * std::numbers::pi / 180.0);
    std::vector<std::array<int, 3>> faces;
    auto emit = [&](int a, int b, int c) {
        // Drop faces inside the polar hole (centroid θ < hole ⇔ z/R large).
        const Eigen::Vector3d cen =
            (Vg.row(a) + Vg.row(b) + Vg.row(c)).transpose() / 3.0;
        if (cen.z() / R > cos_hole) return;            // in the hole cap
        // Outward winding: flip if (v1-v0)×(v2-v0)·centroid < 0.
        const Eigen::Vector3d v0 = Vg.row(a), v1 = Vg.row(b), v2 = Vg.row(c);
        const Eigen::Vector3d e1 = v1 - v0, e2 = v2 - v0;
        if (e1.cross(e2).dot(cen) < 0.0) std::swap(b, c);
        faces.push_back({a, b, c});
    };
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n - i; ++j) {
            emit(id(i, j), id(i + 1, j), id(i, j + 1));
            if (i + j < n - 1)
                emit(id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
        }

    // Compact to used vertices (orphans would break one-ring spacing).
    std::vector<int> remap(static_cast<std::size_t>(n_grid), -1);
    chladni::mesh::TriMesh m;
    int nv = 0;
    for (auto& f : faces)
        for (int v : f)
            if (remap[static_cast<std::size_t>(v)] < 0)
                remap[static_cast<std::size_t>(v)] = nv++;
    m.V.resize(nv, 3);
    for (int v = 0; v < n_grid; ++v)
        if (remap[static_cast<std::size_t>(v)] >= 0)
            m.V.row(remap[static_cast<std::size_t>(v)]) = Vg.row(v);
    m.F.resize(static_cast<Eigen::Index>(faces.size()), 3);
    for (std::size_t t = 0; t < faces.size(); ++t)
        for (int c = 0; c < 3; ++c)
            m.F(static_cast<Eigen::Index>(t), c) =
                remap[static_cast<std::size_t>(faces[t][c])];

    if (conform_rim) {
        // Project the hole-rim boundary vertices onto the exact
        // θ = theta_hole_deg circle (keep φ). Rim vertices wander in
        // θ ∈ [hole − step, hole + ~0.5 step] on the face-dropped cut;
        // the first NON-rim boundary vertex (down a symmetry meridian)
        // sits a full step below the pinch corner, so a 0.65-step
        // threshold separates the two cleanly at every n.
        const double step_rad = (std::numbers::pi / 2.0) / n;
        const double th_hole  = theta_hole_deg * std::numbers::pi / 180.0;
        const double th_max   = th_hole + 0.65 * step_rad;
        std::vector<char> on_bdry(static_cast<std::size_t>(nv), 0);
        for (const auto& be :
             chladni::shell::lme::collect_boundary_edges(m.F)) {
            on_bdry[static_cast<std::size_t>(be.v0)] = 1;
            on_bdry[static_cast<std::size_t>(be.v1)] = 1;
        }
        for (int v = 0; v < nv; ++v) {
            if (!on_bdry[static_cast<std::size_t>(v)]) continue;
            const Eigen::Vector3d p = m.V.row(v).transpose();
            const double theta = std::acos(p.z() / R);
            if (theta >= th_max) continue;            // not a rim vertex
            const double phi = std::atan2(p.y(), p.x());
            m.V(v, 0) = R * std::sin(th_hole) * std::cos(phi);
            m.V(v, 1) = R * std::sin(th_hole) * std::sin(phi);
            m.V(v, 2) = R * std::cos(th_hole);
        }
    }
    return m;
}

/**
 * @brief Solve the linear elastostatic problem @f$ K U = f @f$ with a set
 *        of homogeneous Dirichlet constraints, returning the full DOF
 *        solution vector.
 *
 * Constrained DOFs are removed (row/column elimination via a selection
 * matrix), the reduced SPD system is factorised with @c SimplicialLDLT,
 * and the solution is scattered back into a full-length vector with zeros
 * on the constrained entries.
 *
 * @param K            Global stiffness, @f$ n_\text{dof} \times n_\text{dof} @f$
 *                     (may exceed @f$ 3 N @f$ when ghost DOFs are present).
 * @param f            Load vector, length @c K.rows() (ghost entries zero).
 * @param constrained  Sorted/unsorted list of DOF indices pinned to 0.
 * @return Full-length solution @f$ U @f$ (length @c K.rows()).
 * @throws std::runtime_error if the reduced factorisation fails (singular
 *         system — typically an unconstrained rigid-body mode).
 */
Eigen::VectorXd solve_static_dirichlet(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::VectorXd&             f,
    const std::vector<Eigen::Index>&   constrained)
{
    const Eigen::Index n = K.rows();

    std::vector<char> is_fixed(static_cast<std::size_t>(n), 0);
    for (const Eigen::Index d : constrained) {
        is_fixed[static_cast<std::size_t>(d)] = 1;
    }
    std::vector<Eigen::Index> free_dofs;
    free_dofs.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index d = 0; d < n; ++d) {
        if (!is_fixed[static_cast<std::size_t>(d)]) {
            free_dofs.push_back(d);
        }
    }
    const Eigen::Index n_free =
        static_cast<Eigen::Index>(free_dofs.size());

    Eigen::SparseMatrix<double> P(n, n_free);
    P.reserve(Eigen::VectorXi::Constant(n_free, 1));
    for (Eigen::Index k = 0; k < n_free; ++k) {
        P.insert(free_dofs[static_cast<std::size_t>(k)], k) = 1.0;
    }
    P.makeCompressed();

    Eigen::SparseMatrix<double> K_red = P.transpose() * K * P;
    K_red.makeCompressed();
    const Eigen::VectorXd f_red = P.transpose() * f;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K_red);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "solve_static_dirichlet: reduced factorisation failed "
            "(singular system — check for unconstrained rigid modes)");
    }
    const Eigen::VectorXd u_red = solver.solve(f_red);

    Eigen::VectorXd U = Eigen::VectorXd::Zero(n);
    for (Eigen::Index k = 0; k < n_free; ++k) {
        U(free_dofs[static_cast<std::size_t>(k)]) = u_red(k);
    }
    return U;
}

/// One pinched-cylinder result: normalised radial deflection under the load.
struct PinchedResult {
    Eigen::Index n_vertices = 0;
    double       w_under_load = 0.0;   ///< |radial displacement| at the load
    double       normalised  = 0.0;    ///< w_under_load / w_ref
};

/// Reference radial deflection under the load (MacNeal-Harder 1985).
constexpr double kPinchedCylRef = 1.8248e-5;

/**
 * @brief Run the pinched-cylinder benchmark on one mesh with one assembler.
 *
 * Builds @c generate_cylinder(R, L, n_around, n_along), assembles @c K,
 * applies rigid-diaphragm BCs (u_x = u_y = 0 on both end rings), pins one
 * node's u_z to remove the residual axial rigid-translation mode, applies
 * two opposite inward radial unit loads at mid-length (θ = 0, π), solves,
 * and returns the radial deflection at the θ = 0 load vertex.
 *
 * @param assembler  Any @ref chladni::shell::ShellAssembler.
 * @param n_around   Vertices per ring (must be even; θ = π load needs i = n_around/2).
 * @param n_along    Axial segments (must be even; mid-length ring needs j = n_along/2).
 */
PinchedResult run_pinched_cylinder(
    const chladni::shell::ShellAssembler& assembler,
    int n_around, int n_along)
{
    constexpr double R = 300.0;
    constexpr double L = 600.0;
    constexpr double t = 3.0;
    constexpr double P_load = 1.0;

    if (n_around % 2 != 0 || n_along % 2 != 0) {
        throw std::invalid_argument(
            "run_pinched_cylinder: n_around and n_along must both be even");
    }

    const auto mesh = chladni::mesh::generate_cylinder(R, L, n_around, n_along);
    const Eigen::Index n_v = mesh.V.rows();

    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(
            pinched_cylinder_material(), t);

    const Eigen::SparseMatrix<double> K =
        assembler.assemble_K(mesh.V, mesh.F, sm);
    const Eigen::Index n_dof = K.rows();   // 3*n_v, or 3*(n_v+G) with ghosts

    // ----- rigid-diaphragm Dirichlet BCs --------------------------------
    // End rings j = 0 and j = n_along: pin u_x and u_y (in-plane), u_z free.
    std::vector<Eigen::Index> constrained;
    auto pin_ring = [&](int j) {
        for (int i = 0; i < n_around; ++i) {
            const Eigen::Index v = static_cast<Eigen::Index>(j) * n_around + i;
            constrained.push_back(3 * v + 0);   // u_x
            constrained.push_back(3 * v + 1);   // u_y
        }
    };
    pin_ring(0);
    pin_ring(n_along);
    // The only remaining rigid mode is uniform axial translation (u_z = c);
    // pin u_z at a single end node to make K_red SPD.
    constrained.push_back(3 * (static_cast<Eigen::Index>(0) * n_around + 0) + 2);

    // ----- two opposite inward radial unit loads at mid-length ----------
    const int  j_mid = n_along / 2;
    const Eigen::Index p0 =
        static_cast<Eigen::Index>(j_mid) * n_around + 0;             // θ = 0  (+x)
    const Eigen::Index p1 =
        static_cast<Eigen::Index>(j_mid) * n_around + n_around / 2;  // θ = π  (-x)

    Eigen::VectorXd f = Eigen::VectorXd::Zero(n_dof);
    f(3 * p0 + 0) = -P_load;   // inward at θ = 0 → -x
    f(3 * p1 + 0) = +P_load;   // inward at θ = π → +x

    const Eigen::VectorXd U = solve_static_dirichlet(K, f, constrained);

    // Radial direction at θ = 0 is +x, so the radial deflection is U_x there.
    PinchedResult out;
    out.n_vertices   = n_v;
    out.w_under_load = std::abs(U(3 * p0 + 0));
    out.normalised   = out.w_under_load / kPinchedCylRef;
    return out;
}

}  // namespace

/// Timoshenko & Woinowsky-Krieger coefficients for a simply-supported
/// square plate of side @c a, flexural rigidity @c D:
///   - central point load @c P:        w_max = 0.01160 P a^2 / D
///   - uniform pressure   @c q:         w_max = 0.00406 q a^4 / D
constexpr double kSSPlatePointCoeff   = 0.01160;
constexpr double kSSPlateUniformCoeff = 0.00406;

/**
 * @brief Harness validation: simply-supported square plate, central
 *        deflection vs the exact Kirchhoff coefficients, under BOTH a
 *        central point load and a uniform pressure.
 *
 * This isolates the static-solve machinery (BC, load, factorisation,
 * read-out) and the plate BENDING operator — a flat plate decouples
 * in-plane from transverse, so there is no membrane participation.
 *
 * The two load types discriminate a meshfree subtlety: LME is
 * non-interpolating, so dumping a point load on one coefficient DOF (and
 * reading one coefficient DOF) only approximates a true point
 * load/displacement via the weak Kronecker-delta property — exact only at
 * the boundary, approximate in the interior. A uniform pressure applied
 * as the *consistent* nodal force @f$ f_a = q\int p_a\,dA @f$ (the
 * mass-matrix row-sum) has no such concentration, so it cleanly tests the
 * operator. If LME matches the analytic deflection under pressure but not
 * under the point load, the discrepancy is point-load representation, not
 * the bending operator.
 *
 * BCs: pin every in-plane DOF (u_x, u_y) to kill the decoupled in-plane
 * null space, and pin u_z = 0 on the boundary (simple support). Ghost
 * nodes OFF (a flat plate has no boundary curvature to flatten).
 */
TEST_CASE("simply-supported square plate: harness + bending vs Kirchhoff, "
          "point load vs uniform pressure (Loop and LME)",
          "[shell][static][harness][ss_plate]")
{
    constexpr double a = 1.0;      // square side (m)
    constexpr double P_load = 1.0; // central point load (N)
    constexpr double q_load = 1.0; // uniform pressure (N/m^2)
    constexpr double E  = 2.0e11;
    constexpr double nu = 0.30;
    constexpr double h  = 1.0e-3;
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));   // = k_B
    const double w_ref_pt  = kSSPlatePointCoeff   * P_load * a * a / D;
    const double w_ref_uni = kSSPlateUniformCoeff * q_load * a * a * a * a / D;

    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(
            {.youngs_modulus = E, .poisson_ratio = nu, .density = 1.0}, h);

    std::printf("\n=== Simply-supported square plate (a=1) ===\n");
    std::printf("analytic w(point)   = %.6e (alpha=0.01160)\n", w_ref_pt);
    std::printf("analytic w(uniform) = %.6e (alpha=0.00406)\n", w_ref_uni);
    std::printf("%-18s %6s %12s %10s %12s %10s\n", "method", "ncell",
                "w_pt", "pt/ref", "w_uni", "uni/ref");

    struct Ratios { double pt; double uni; };
    auto run = [&](const char* name,
                   const chladni::shell::ShellAssembler& A,
                   double rho_h_for_M, int ncell) -> Ratios {
        const auto mesh = chladni::mesh::generate_flat_plate(a, a, ncell, ncell);
        const Eigen::Index n_v = mesh.V.rows();
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::SparseMatrix<double> M = A.assemble_M(mesh.V, mesh.F, rho_h_for_M);
        const Eigen::Index n_dof = K.rows();

        constexpr double tol = 1e-9;
        Eigen::Index centre = -1;
        double best = 1e30;
        std::vector<Eigen::Index> constrained;
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const double x = mesh.V(v, 0);
            const double y = mesh.V(v, 1);
            constrained.push_back(3 * v + 0);   // pin all u_x
            constrained.push_back(3 * v + 1);   // pin all u_y
            const bool on_bdry = (std::abs(x) < tol || std::abs(x - a) < tol ||
                                  std::abs(y) < tol || std::abs(y - a) < tol);
            if (on_bdry) constrained.push_back(3 * v + 2);   // w = 0
            const double d2 = (x - 0.5 * a) * (x - 0.5 * a) +
                              (y - 0.5 * a) * (y - 0.5 * a);
            if (d2 < best) { best = d2; centre = v; }
        }

        // Point load: all of P on the centre z-DOF.
        Eigen::VectorXd f_pt = Eigen::VectorXd::Zero(n_dof);
        f_pt(3 * centre + 2) = -P_load;
        const Eigen::VectorXd U_pt = solve_static_dirichlet(K, f_pt, constrained);
        const double w_pt = std::abs(U_pt(3 * centre + 2));

        // Uniform pressure: consistent nodal force f_a = q * ∫ p_a dA, the
        // z-block row-sum of M (assembled with rho_h = 1 → M_ab = ∫ p_a p_b).
        Eigen::VectorXd e_z = Eigen::VectorXd::Zero(n_dof);
        for (Eigen::Index v = 0; v < n_v; ++v) e_z(3 * v + 2) = 1.0;
        Eigen::VectorXd f_uni = -q_load * (M * e_z);   // downward
        // Zero out in-plane rows defensively (M decouples, but be explicit).
        for (Eigen::Index v = 0; v < n_v; ++v) {
            f_uni(3 * v + 0) = 0.0;
            f_uni(3 * v + 1) = 0.0;
        }
        const Eigen::VectorXd U_uni = solve_static_dirichlet(K, f_uni, constrained);
        const double w_uni = std::abs(U_uni(3 * centre + 2));

        // Compliance fᵀU under the distributed load: a basis-invariant
        // Galerkin energy (independent of the pointwise coefficient→
        // displacement read). Comparing this across methods isolates the
        // OPERATOR from the read-out.
        const double compliance_uni = f_uni.dot(U_uni);

        std::printf("%-18s %6d %12.5e %10.4f %12.5e %10.4f  C=%.5e\n",
                    name, ncell, w_pt, w_pt / w_ref_pt, w_uni,
                    w_uni / w_ref_uni, compliance_uni);
        return {w_pt / w_ref_pt, w_uni / w_ref_uni};
    };

    chladni::shell::LMEAssembler::Params lme_p;
    lme_p.use_ghost_nodes = false;       // flat plate: ghosts unnecessary
    chladni::shell::LMEAssembler lme{lme_p};
    chladni::shell::LoopAssembler loop;

    for (const int nc : {16, 24, 32}) {
        const Ratios loop_r = run("Loop", loop, 1.0, nc);
        const Ratios lme_r  = run("LME (ghost-off)", lme, 1.0, nc);
        // Loop bending is validated elsewhere (Euler-Bernoulli strip); gate
        // it near the analytic coefficients here as a harness sanity check.
        CHECK(loop_r.uni == Catch::Approx(1.0).margin(0.10));
        CHECK(std::isfinite(lme_r.pt));
        CHECK(std::isfinite(lme_r.uni));
        CHECK(lme_r.uni > 0.0);
    }
}

/**
 * @brief Diagnostic: prescribed flat-plate bending field, direct operator
 *        energy vs analytic — no eigensolver, no static solve, no read.
 *
 * Prescribe @f$ w = \sin(\pi x/a)\sin(\pi y/a) @f$ (u_x = u_y = 0) at every
 * vertex and evaluate the bending energy @f$ \tfrac12 u^\top K_\text{bend} u @f$
 * (assembled with @c k_L = 0). For this shape on a simply-supported unit
 * square the exact Kirchhoff bending energy is @f$ D\,\pi^4/2 @f$. This is
 * the purest probe of the bending OPERATOR: it removes the eigensolver,
 * the static solve, AND the coefficient→displacement read, so any LME/Loop
 * deviation is the operator's stiffness itself.
 */
TEST_CASE("LME flat-plate bending operator — prescribed-field energy vs "
          "analytic (curved vs flat path)",
          "[shell][static][harness][ss_plate_bend_energy][.diag]")
{
    constexpr double a = 1.0;
    constexpr double E = 2.0e11, nu = 0.30, h = 1.0e-3;
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));
    const double U_analytic = D * std::pow(std::numbers::pi, 4) / 2.0;
    const chladni::shell::ShellMaterial sm_bend =
        chladni::shell::ShellMaterial{
            .k_L = 0.0,
            .k_B = chladni::shell::shell_material_from_isotropic(
                       {.youngs_modulus = E, .poisson_ratio = nu,
                        .density = 1.0}, h).k_B,
            .poisson_ratio = nu};

    constexpr int ncell = 24;
    const auto mesh = chladni::mesh::generate_flat_plate(a, a, ncell, ncell);
    const Eigen::Index n_v = mesh.V.rows();

    // Prescribed bending field on the REAL vertices.
    auto prescribe = [&](Eigen::Index n_dof) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(n_dof);
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const double x = mesh.V(v, 0), y = mesh.V(v, 1);
            u(3 * v + 2) = std::sin(std::numbers::pi * x / a) *
                           std::sin(std::numbers::pi * y / a);
        }
        return u;   // ghost DOFs (>= 3*n_v) left zero
    };

    std::printf("\n=== Prescribed flat-plate bending field "
                "w=sin(pi x)sin(pi y), ncell=24 ===\n");
    std::printf("analytic bending energy D*pi^4/2 = %.5e\n", U_analytic);

    auto eval = [&](const char* name, const chladni::shell::ShellAssembler& A) {
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm_bend);
        const Eigen::VectorXd u = prescribe(K.rows());
        const double Eb = 0.5 * u.dot(K * u);
        std::printf("%-26s E_bend=%.5e  E/analytic=%7.4f\n",
                    name, Eb, Eb / U_analytic);
    };

    // HISTORY (2026-05-29): the curved path once used a fixed
    // r_cut_mult_curved=1.4 that STARVED the bending Hessian here (0.639×
    // analytic, rising only to ~0.97× by r_cut≈3). It now uses the paper's
    // value-based neighbour cutoff (Params::tol_lme, Millán Eq. 2), which
    // resolves bending without per-mesh tuning. The three rows below
    // confirm Loop, the flat LME path, and the (fixed) curved LME default
    // all give ~0.97-1.0× analytic bending energy.
    { chladni::shell::LoopAssembler loop; eval("Loop", loop); }
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_curved_shell = false;
        p.use_ghost_nodes  = false;
        chladni::shell::LMEAssembler lme{p};
        eval("LME flat path", lme);
    }
    {
        chladni::shell::LMEAssembler::Params p;   // shipped default: curved
        p.use_ghost_nodes  = false;               // ghost-off so u=3*n_v
        chladni::shell::LMEAssembler lme{p};
        eval("LME curved (value-based)", lme);
    }
}

/**
 * @brief Diagnostic: localise the LME flat-plate-bending softness across
 *        config knobs (ghost on/off × curved on/off), under the
 *        consistent uniform-pressure load.
 *
 * Loop matches the exact Kirchhoff coefficient, so the LME/Loop deflection
 * and compliance ratios measure the LME operator's softness directly. The
 * uniform-pressure compliance @c fᵀU is basis-invariant, so it isolates
 * the operator from the coefficient→displacement read.
 */
TEST_CASE("LME flat-plate bending softness — config sweep (ghost/curved)",
          "[shell][static][harness][ss_plate_config][.diag]")
{
    constexpr double a = 1.0, q_load = 1.0;
    constexpr double E = 2.0e11, nu = 0.30, h = 1.0e-3;
    const double D = E * h * h * h / (12.0 * (1.0 - nu * nu));
    const double w_ref_uni = kSSPlateUniformCoeff * q_load * a * a * a * a / D;
    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(
            {.youngs_modulus = E, .poisson_ratio = nu, .density = 1.0}, h);
    constexpr int ncell = 24;
    const auto mesh = chladni::mesh::generate_flat_plate(a, a, ncell, ncell);
    const Eigen::Index n_v = mesh.V.rows();

    constexpr double tol = 1e-9;
    Eigen::Index centre = -1;
    double best = 1e30;
    std::vector<Eigen::Index> base_constrained;
    for (Eigen::Index v = 0; v < n_v; ++v) {
        const double x = mesh.V(v, 0), y = mesh.V(v, 1);
        base_constrained.push_back(3 * v + 0);
        base_constrained.push_back(3 * v + 1);
        const bool on_bdry = (std::abs(x) < tol || std::abs(x - a) < tol ||
                              std::abs(y) < tol || std::abs(y - a) < tol);
        if (on_bdry) base_constrained.push_back(3 * v + 2);
        const double d2 = (x - 0.5 * a) * (x - 0.5 * a) +
                          (y - 0.5 * a) * (y - 0.5 * a);
        if (d2 < best) { best = d2; centre = v; }
    }

    auto eval = [&](const char* name, const chladni::shell::ShellAssembler& A) {
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::SparseMatrix<double> M = A.assemble_M(mesh.V, mesh.F, 1.0);
        const Eigen::Index n_dof = K.rows();
        Eigen::VectorXd e_z = Eigen::VectorXd::Zero(n_dof);
        for (Eigen::Index v = 0; v < n_v; ++v) e_z(3 * v + 2) = 1.0;
        Eigen::VectorXd f = -q_load * (M * e_z);
        for (Eigen::Index v = 0; v < n_v; ++v) { f(3*v+0)=0.0; f(3*v+1)=0.0; }
        const Eigen::VectorXd U = solve_static_dirichlet(K, f, base_constrained);
        const double w = std::abs(U(3 * centre + 2));
        const double C = f.dot(U);
        std::printf("%-26s w/wref=%8.4f  C=%.5e\n", name, w / w_ref_uni, C);
    };

    std::printf("\n=== LME flat-plate bending config sweep (uniform load, "
                "ncell=24) ===\n");
    { chladni::shell::LoopAssembler loop; eval("Loop (reference)", loop); }
    for (const bool curved : {true, false}) {
        for (const bool ghost : {true, false}) {
            chladni::shell::LMEAssembler::Params p;
            p.use_curved_shell = curved;
            p.use_ghost_nodes  = ghost;
            chladni::shell::LMEAssembler lme{p};
            char lbl[64];
            std::snprintf(lbl, sizeof lbl, "LME curved=%d ghost=%d",
                          curved ? 1 : 0, ghost ? 1 : 0);
            eval(lbl, lme);
        }
    }
}

/// Scordelis-Lo roof reference vertical deflection at the free-edge
/// midpoint (Belytschko et al. [27]; the paper's overkill value is
/// 0.300575).
constexpr double kScordelisLoRef = 0.3024;

/**
 * @brief Scordelis-Lo roof — the paper's OWN reported cylindrical
 *        benchmark (Millán 2011 §4.3), reproduced as the decisive
 *        "matches the paper" check before any operator change.
 *
 * Standard parameters: R=25, L=50, t=0.25, E=4.32e8, ν=0, self-weight
 * g=90 per unit area (vertical, −z). Rigid diaphragms at the two curved
 * ends (u_y=u_z=0, u_x free); the two straight edges are free. The
 * reported quantity is the vertical deflection at the midpoint of a free
 * edge: reference 0.3024 (paper overkill 0.300575).
 *
 * This benchmark is ideal for LME: the load is DISTRIBUTED (consistent
 * body force f_a = −g ∫ p_a dA, the z-block row-sum of M — no point-load
 * pathology), and the deflection is read at a FREE-EDGE vertex where the
 * LME weak Kronecker-delta property is exact (coefficient = displacement).
 *
 * We run Loop and LME at the bending-correct WIDE stencil
 * (r_cut_mult_curved ≈ 4), plus the default narrow 1.4 for contrast — the
 * narrow stencil under-resolves bending and should miss the reference,
 * while the wide stencil should reproduce it (Scordelis-Lo is membrane+
 * bending dominated, NOT inextensional, so membrane locking does not bite
 * here — exactly why the paper's cylindrical test passes).
 */
TEST_CASE("Scordelis-Lo roof (paper §4.3): Loop & LME vs reference 0.3024",
          "[shell][static][obstacle][scordelis_lo][.diag]")
{
    constexpr double R = 25.0, L = 50.0, t = 0.25;
    constexpr double E = 4.32e8, nu = 0.0, g = 90.0;
    constexpr double half_deg = 40.0;

    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(
            {.youngs_modulus = E, .poisson_ratio = nu, .density = 1.0}, t);

    std::printf("\n=== Scordelis-Lo roof (R=25 L=50 t=0.25 E=4.32e8 nu=0, "
                "g=90/area, diaphragm ends) ===\n");
    std::printf("reference free-edge vertical deflection = %.5f "
                "(paper overkill 0.300575)\n", kScordelisLoRef);
    std::printf("%-26s %8s %12s %10s\n", "method", "n_vert", "w_edge", "w/wref");

    auto run = [&](const char* name,
                   const chladni::shell::ShellAssembler& A,
                   int n_arc, int n_axial) {
        const auto mesh = build_cylindrical_roof(R, L, half_deg, n_arc, n_axial);
        const Eigen::Index n_v = mesh.V.rows();
        const int per_ring = n_arc + 1;
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::SparseMatrix<double> M = A.assemble_M(mesh.V, mesh.F, 1.0);
        const Eigen::Index n_dof = K.rows();

        // Diaphragm BCs: u_y = u_z = 0 on both end rings (u_x free).
        std::vector<Eigen::Index> constrained;
        auto pin_ring = [&](int j) {
            for (int i = 0; i < per_ring; ++i) {
                const Eigen::Index v =
                    static_cast<Eigen::Index>(j) * per_ring + i;
                constrained.push_back(3 * v + 1);   // u_y
                constrained.push_back(3 * v + 2);   // u_z
            }
        };
        pin_ring(0);
        pin_ring(n_axial);
        // Remove the residual axial (u_x) rigid translation: pin one node.
        constrained.push_back(3 * 0 + 0);

        // Self-weight: consistent body force f_a = -g ∫ p_a dA, the z-block
        // row-sum of M (assembled with rho_h=1 → M_ab=∫p_a p_b; PoU gives
        // ∫p_a dA). e_z = 1 on ALL z-DOFs (real + ghost).
        Eigen::VectorXd e_z = Eigen::VectorXd::Zero(n_dof);
        for (Eigen::Index d = 2; d < n_dof; d += 3) e_z(d) = 1.0;
        const Eigen::VectorXd f = -g * (M * e_z);

        const Eigen::VectorXd U = solve_static_dirichlet(K, f, constrained);

        // Free-edge midpoint: i=0 (φ=-40°), j=n_axial/2. Vertical = z.
        const Eigen::Index v_edge =
            static_cast<Eigen::Index>(n_axial / 2) * per_ring + 0;
        const double w = std::abs(U(3 * v_edge + 2));
        std::printf("%-26s %8lld %12.5f %10.4f\n", name,
                    static_cast<long long>(n_v), w, w / kScordelisLoRef);
        return w / kScordelisLoRef;
    };

    // The curved LME path now uses the paper's value-based neighbour
    // cutoff (Params::tol_lme, Millán Eq. 2), so r_cut_mult_curved is inert
    // here — the SHIPPED DEFAULT LME reproduces the paper. (Before the
    // 2026-05-29 fix the fixed r_cut_mult_curved=1.4 gave ~1.22× — bending
    // starved; that knob no longer affects the 1st-order LME curved path.)
    double loop_fine = 0.0, lme_fine = 0.0, sme_fine = 0.0;
    for (const int n : {16, 24, 32}) {
        chladni::shell::LoopAssembler loop;
        loop_fine = run("Loop", loop, n, n);

        chladni::shell::LMEAssembler lme{chladni::shell::LMEAssembler::Params{}};
        lme_fine = run("LME (shipped default)", lme, n, n);

        // SME faithfulness check: Scordelis-Lo is membrane+bending
        // dominated (NOT inextensional), so it does not probe locking —
        // it tests whether the SME shell assembly reproduces the paper's
        // own cylindrical benchmark at all. A faithful SME should match
        // the reference like LME does.
        chladni::shell::LMEAssembler::Params psme;
        psme.use_second_order_sme = true;
        chladni::shell::LMEAssembler sme{psme};
        sme_fine = run("SME (2nd-order, defaults)", sme, n, n);

        // SME with the PAPER's parameters (Rosolen-Millán-Arroyo 2013:
        // γ_LME=0.8, α∈{1.6,2,2.5}) — our defaults (γ=1.6, α=4) are OUTSIDE
        // the paper's tested range. If paper-params SME matches the
        // reference, SME is faithful and the defaults are the culprit.
        chladni::shell::LMEAssembler::Params ppap;
        ppap.use_second_order_sme = true;
        ppap.gamma     = 0.8;
        ppap.sme_alpha = 2.0;
        chladni::shell::LMEAssembler sme_pap{ppap};
        try {
            run("SME (paper γ=0.8 α=2)", sme_pap, n, n);
        } catch (const std::exception& e) {
            std::printf("%-26s %8d %12s %10s  (%s)\n",
                "SME (paper γ=0.8 α=2)", (n + 1) * (n + 1),
                "DIVERGED", "-", e.what());
        }
    }

    // Paper-match gates at the finest mesh (1089 vertices): Loop and the
    // shipped-default LME reproduce the Scordelis-Lo reference (the paper's
    // own overkill value is itself 0.6% low). Guards the value-based-
    // truncation bending fix.
    CHECK(loop_fine == Catch::Approx(1.0).margin(0.05));
    CHECK(lme_fine  == Catch::Approx(1.0).margin(0.06));

    // SME FAITHFULNESS — RESOLVED 2026-06. At the paper's α=2 default,
    // faithful SME now reproduces Scordelis-Lo: w/wref ≈ 1.10 → 1.03 →
    // 1.01 over 289 → 625 → 1089 V, converging to the reference and
    // BEATING 1st-order LME (1.06 → 1.02). The earlier "15-20% too soft"
    // reading was a compound artifact of (a) an α=4 default forced by a
    // Newton convergence-criterion bug that mis-reported feasible α=2
    // solves as divergences, and (b) the chart-rim-zeroing hack masking a
    // boundary-gap over-stiffness. Fixes: gradient-scaled ridge +
    // objective-stagnation convergence + relative tol in the SME Newton;
    // the faithful §3.2.2 graded boundary-gap recipe (replacing the hack);
    // and α=2 default. See [[chladni-stiffness-alternatives]] §f.
    CHECK(sme_fine == Catch::Approx(1.0).margin(0.06));
}

/**
 * @brief Pinched hemisphere (MacNeal-Harder / Rosolen-Millán-Arroyo 2013
 *        §4.4, Fig 17) — the paper's OWN inextensional curved-shell SME
 *        benchmark, the decisive faithfulness check for the curved path.
 *
 * R=10, t=0.04, E=6.825e7, ν=0.3, 18° polar hole, free edges. Two diametral
 * radial point loads (outward at φ=0, inward at φ=90). Quarter-symmetry
 * model (φ∈[0,90]): u_y=0 on the φ=0 plane, u_x=0 on the φ=90 plane, one
 * u_z pin to kill the z rigid translation. Reference radial displacement at
 * the load point δ_r = 0.0924 (paper overkill 0.092401).
 *
 * This assesses representation of INEXTENSIONAL deformation with curvature
 * in two directions — the SAME regime that locks our free-free cylinder.
 * The paper shows BOTH LME(γ=0.8) and SME(α=1.6,2,2.5) converging here, so
 * if OUR LME/SME lock relative to Loop, the curved path has a real defect.
 */
TEST_CASE("pinched hemisphere (paper §4.4 Fig17): Loop vs LME vs SME radial "
          "displacement vs ref 0.0924",
          "[shell][static][obstacle][hemisphere][.diag][.slow]")
{
    constexpr double R = 10.0, t = 0.04, hole_deg = 18.0;
    constexpr double ref = 0.0924;
    constexpr double F = 1.0;     // P/2 per symmetry-plane load point (P=2)
    const auto mat = pinched_hemisphere_material();
    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(mat, t);

    std::printf("\n=== Pinched hemisphere (R=10 t=0.04 E=6.825e7 ν=0.3, "
                "18° hole, diametral loads F=%.1f) ===\n", F);
    std::printf("reference radial displacement at load = %.5f\n", ref);
    std::printf("%-26s %8s %12s %10s\n", "method", "n_vert", "δ_r", "δ/δref");

    auto run = [&](const char* name,
                   const chladni::shell::ShellAssembler& A,
                   int n_theta, int n_phi) -> double {
        const auto mesh =
            build_quarter_hemisphere(R, hole_deg, n_theta, n_phi);
        const int per_row = n_phi + 1;
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::Index n_dof = K.rows();

        // Symmetry + rigid-body constraints (real-vertex DOFs).
        std::vector<Eigen::Index> constrained;
        for (int j = 0; j <= n_theta; ++j) {
            const Eigen::Index v0 = static_cast<Eigen::Index>(j) * per_row + 0;
            constrained.push_back(3 * v0 + 1);             // φ=0: u_y=0
            const Eigen::Index v9 =
                static_cast<Eigen::Index>(j) * per_row + n_phi;
            constrained.push_back(3 * v9 + 0);             // φ=90: u_x=0
        }
        constrained.push_back(3 * 0 + 2);                  // z-pin at hole/φ=0

        // Diametral radial point loads on the equator (j=n_theta).
        Eigen::VectorXd f = Eigen::VectorXd::Zero(n_dof);
        const Eigen::Index v_out =
            static_cast<Eigen::Index>(n_theta) * per_row + 0;       // φ=0
        const Eigen::Index v_in =
            static_cast<Eigen::Index>(n_theta) * per_row + n_phi;   // φ=90
        f(3 * v_out + 0) += F;     // outward radial = +x at φ=0
        f(3 * v_in  + 1) += -F;    // inward  radial = −y at φ=90

        const Eigen::VectorXd U = solve_static_dirichlet(K, f, constrained);
        const double dr = U(3 * v_out + 0);   // radial = u_x at φ=0 equator
        std::printf("%-26s %8lld %12.6f %10.4f\n", name,
                    static_cast<long long>(mesh.V.rows()), dr, dr / ref);
        return dr / ref;
    };

    for (const auto lv : {std::array<int,2>{8, 8},
                          std::array<int,2>{12, 12},
                          std::array<int,2>{16, 16},
                          std::array<int,2>{24, 24},
                          std::array<int,2>{32, 32}}) {
        const int nt = lv[0], np = lv[1];
        chladni::shell::LoopAssembler loop;
        run("Loop", loop, nt, np);

        chladni::shell::LMEAssembler lme_d{chladni::shell::LMEAssembler::Params{}};
        try { run("LME (default γ=1.6)", lme_d, nt, np); }
        catch (const std::exception& e) { std::printf("  LME default DIVERGED: %s\n", e.what()); }

        chladni::shell::LMEAssembler::Params plme; plme.gamma = 0.8;
        chladni::shell::LMEAssembler lme_p{plme};
        try { run("LME (paper γ=0.8)", lme_p, nt, np); }
        catch (const std::exception& e) { std::printf("  LME paper DIVERGED: %s\n", e.what()); }

        // SME (default α=2) on this STRUCTURED LAT-LONG grid throws: the
        // azimuthal spacing R·sinθ·dφ collapses toward the hole, producing
        // anisotropic sliver triangles whose wPCA charts are genuinely
        // 2nd-order-INFEASIBLE (0 ∉ conv{(x_a, x_a⊗x_a − d_a)}) at all α.
        // This is a MESH-QUALITY mismatch, NOT a solver/recipe defect:
        // max-ent is a scattered/quality-point method, and on the paper's
        // OWN geodesic (subdivided-octahedron) discretization SME converges
        // at α=2 — see the [hemisphere_geo] test, which reproduces Fig 17.
        // Scordelis-Lo / closed sphere / polar disk SME all converge at α=2.
        chladni::shell::LMEAssembler::Params psd; psd.use_second_order_sme = true;
        chladni::shell::LMEAssembler sme_d{psd};
        try { run("SME (default α=2)", sme_d, nt, np); }
        catch (const std::exception& e) { std::printf("  SME default DIVERGED: %s\n", e.what()); }
        std::printf("\n");
    }
    SUCCEED("pinched-hemisphere convergence printed to stderr");
}

/**
 * @brief Pinched hemisphere on the PAPER'S OWN discretization — a geodesic
 *        (subdivided-octahedron) point set, per Millán-Rosolen-Arroyo 2011
 *        §2.6. This is the faithful reproduction of Rosolen-Millán-Arroyo
 *        2013 Fig 17: the paper runs SME at α∈{1.6, 2, 2.5} and LME at
 *        γ=0.8 and BOTH converge (slope ≈3) to δ_r = 0.0924.
 *
 * The lat-long sibling test above feeds SME a structured anisotropic grid
 * (sliver triangles near the hole) — degenerate for a meshfree max-ent
 * scheme — and SME's 2nd-order feasibility genuinely fails on it. On the
 * geodesic point set the method actually targets, SME converges at the
 * paper's α=2, exactly as our isotropic icosphere does. This isolates the
 * earlier "hemisphere divergence" as a MESH-QUALITY mismatch with the
 * paper, not a solver/recipe defect.
 *
 * Same physics (R=10, t=0.04, 18° hole, two diametral radial loads,
 * quarter-symmetry), but vertices are selected GEOMETRICALLY since the
 * geodesic mesh has no (j,i) structure: y=0 plane → u_y=0, x=0 plane →
 * u_x=0, load corners at (R,0,0) [+x out] and (0,R,0) [−y in].
 */
TEST_CASE("pinched hemisphere (paper §4.4, GEODESIC mesh): SME α=1.6/2/2.5 "
          "converges like the paper",
          "[shell][static][obstacle][hemisphere_geo][.diag][.slow]")
{
    constexpr double R = 10.0, t = 0.04, hole_deg = 18.0;
    constexpr double ref = 0.0924, F = 1.0;
    const auto mat = pinched_hemisphere_material();
    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(mat, t);

    std::printf("\n=== Pinched hemisphere, GEODESIC mesh (paper Fig17 "
                "discretization) ===\n");
    std::printf("reference δ_r = %.5f\n", ref);
    std::printf("%-28s %8s %12s %10s\n", "method", "n_vert", "δ_r", "δ/δref");

    auto run = [&](const char* name,
                   const chladni::shell::ShellAssembler& A, int n) -> double {
        const auto mesh = build_quarter_hemisphere_geodesic(R, hole_deg, n);
        const Eigen::Index n_v = mesh.V.rows();
        const double tol = 1.0e-6 * R;
        auto find = [&](const Eigen::Vector3d& p) {
            Eigen::Index best = 0; double bd = 1e30;
            for (Eigen::Index v = 0; v < n_v; ++v) {
                const double d = (mesh.V.row(v).transpose() - p).norm();
                if (d < bd) { bd = d; best = v; }
            }
            return best;
        };
        const Eigen::Index v_out = find({R, 0, 0});   // equator ∩ φ=0
        const Eigen::Index v_in  = find({0, R, 0});   // equator ∩ φ=90

        double dr = std::numeric_limits<double>::quiet_NaN();
        try {
            const Eigen::SparseMatrix<double> K =
                A.assemble_K(mesh.V, mesh.F, sm);
            std::vector<Eigen::Index> constrained;
            for (Eigen::Index v = 0; v < n_v; ++v) {
                if (std::abs(mesh.V(v, 1)) < tol) constrained.push_back(3 * v + 1); // φ=0
                if (std::abs(mesh.V(v, 0)) < tol) constrained.push_back(3 * v + 0); // φ=90
            }
            constrained.push_back(3 * v_out + 2);     // z-pin (kills u_z RBM)

            Eigen::VectorXd f = Eigen::VectorXd::Zero(K.rows());
            f(3 * v_out + 0) += F;     // outward radial +x
            f(3 * v_in  + 1) += -F;    // inward  radial −y

            const Eigen::VectorXd U = solve_static_dirichlet(K, f, constrained);
            dr = U(3 * v_out + 0);
            std::printf("%-28s %8lld %12.6f %10.4f\n", name,
                        static_cast<long long>(n_v), dr, dr / ref);
        } catch (const std::exception& e) {
            std::printf("%-28s %8lld %12s %10s  (%s)\n", name,
                        static_cast<long long>(n_v), "n/a", "-", e.what());
        }
        return dr / ref;
    };

    double sme_fine = std::numeric_limits<double>::quiet_NaN();
    for (const int n : {16, 24, 32, 48, 64}) {
        chladni::shell::LoopAssembler loop;
        run("Loop", loop, n);

        chladni::shell::LMEAssembler::Params plme; plme.gamma = 0.8;
        run("LME (paper γ=0.8)", chladni::shell::LMEAssembler{plme}, n);

        chladni::shell::LMEAssembler::Params psme;
        psme.use_second_order_sme = true; psme.gamma = 0.8; psme.sme_alpha = 2.0;
        sme_fine = run("SME (paper γ=0.8 α=2)",
                       chladni::shell::LMEAssembler{psme}, n);

        chladni::shell::LMEAssembler::Params psme25;
        psme25.use_second_order_sme = true; psme25.gamma = 0.8; psme25.sme_alpha = 2.5;
        run("SME (α=2.5)", chladni::shell::LMEAssembler{psme25}, n);
        std::printf("\n");
    }

    // Faithfulness gate. On the paper's OWN (geodesic) discretization, SME
    // at the paper's α=2 runs at every resolution and sits on a flat
    // δ/δref ≈ 1.10–1.12 plateau from the coarsest level (142 V), beside
    // LME γ=0.8's 1.43 → 1.18 ladder — reproducing Fig 17's regime.
    //
    // HISTORY: the 2026-06-02 revision measured a 3.4 → 1.33 convergence
    // ladder here (pre-faithful-quadrature). The 12-pt rule (Q-D2,
    // 2026-06-03) then probed Gauss points near triangle corners and made
    // the coarse levels MARGINALLY infeasible at α=2 (assembly threw,
    // α=2.5 survived); per-patch α escalation (2026-06-04,
    // Params::sme_alpha_escalation_steps) rescues those few points with
    // locally enlarged gaps and the α=2 column now reads ~1.10 at every
    // level — better than globally enlarging slack to α=2.5 (1.37 coarse).
    CHECK(std::isfinite(sme_fine));   // feasible at every level
    CHECK(sme_fine < 1.2);            // on the Fig-17 plateau
}

TEST_CASE("SME alpha escalation rescues marginal feasibility "
          "(geodesic pinched hemisphere, 142 V)",
          "[shell][sme][alpha_escalation]")
{
    // The faithful 12-pt quadrature (Q-D2, 2026-06-03) probes Gauss
    // points near triangle corners, where the coarse geodesic
    // hemisphere's SME charts at the paper's α=2 are MARGINALLY
    // infeasible: at some quadrature points every Shepard-active
    // patch fails and assemble_K throws ("every Shepard-active
    // patch's in-chart Newton diverged"). Per-patch α escalation
    // retries a failing patch with enlarged gap matrices
    // (α_eff = α·2^k, bounded) before the drop-net — gap ENLARGEMENT
    // is the paper's own feasibility mechanism (RMA13 §3.2–3.3) and α
    // its free regularisation parameter, which the paper already
    // grades spatially near boundaries.
    constexpr double R = 10.0, t = 0.04, hole_deg = 18.0;
    const auto mat = pinched_hemisphere_material();
    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(mat, t);
    const auto mesh = build_quarter_hemisphere_geodesic(R, hole_deg, 16);

    chladni::shell::LMEAssembler::Params p;
    p.use_second_order_sme = true;
    p.gamma     = 0.8;
    p.sme_alpha = 2.0;

    // Escalation OFF reproduces the marginal infeasibility — guards
    // that this fixture actually exercises the mechanism (if the
    // fixture ever becomes feasible outright, this arm flags it and
    // the test needs a harder fixture).
    {
        auto p0 = p;
        p0.sme_alpha_escalation_steps = 0;
        chladni::shell::LMEAssembler asm0{p0};
        REQUIRE_THROWS(asm0.assemble_K(mesh.V, mesh.F, sm));
    }

    // Escalation ON (shipped default) assembles.
    chladni::shell::LMEAssembler asm_{p};
    Eigen::SparseMatrix<double> K;
    REQUIRE_NOTHROW(K = asm_.assemble_K(mesh.V, mesh.F, sm));
    REQUIRE(K.rows() == 3 * (mesh.V.rows()
                             + static_cast<Eigen::Index>(
                                   chladni::shell::lme::
                                       collect_boundary_edges(mesh.F)
                                           .size())));

    // B1-dormancy gate (2026-06-07 pinch-corner dissection,
    // [sme_hemi_dissect] + [hemi_corner_cls]): the marginal queries are
    // Gauss points ~0.1 h from the GENUINE hole/symmetry pinch-corner
    // nodes, whose §3.2.2 gap is d=0; the obstruction is certified
    // 2nd-order (min-norm separation 8e-3) and one B3 ×2 step cures it.
    // Escalation must therefore (a) fire (else this fixture stopped
    // exercising the mechanism and needs replacing) and (b) fully
    // absorb the failures — the B1 drop-net stays DORMANT here, as it
    // is on the cylinder since interior ownership (6214ed0).
    const auto& ds = asm_.last_drop_stats();
    CHECK(ds.n_escal_ok > 0);
    CHECK(ds.n_newton == 0);
    CHECK(ds.w_newton == 0.0);

    // Paper-discretization arm ([hemi_rim_snap] A/B, 2026-06-07): on
    // the CONFORMAL (rim-snapped) cut — the paper's clean-rim
    // discretization — the §3.2.2 recipe at α=2 is feasible OUTRIGHT:
    // no escalation, no drops, even with the ladder disabled. The
    // marginality above is the jagged face-dropped wedge cloud, not
    // the recipe; B3 is the robustness layer for non-paper meshes.
    {
        const auto mesh_c =
            build_quarter_hemisphere_geodesic(R, hole_deg, 16, true);
        auto p0 = p;
        p0.sme_alpha_escalation_steps = 0;
        chladni::shell::LMEAssembler asm_c{p0};
        REQUIRE_NOTHROW(asm_c.assemble_K(mesh_c.V, mesh_c.F, sm));
        const auto& dsc = asm_c.last_drop_stats();
        CHECK(dsc.n_escal_ok == 0);
        CHECK(dsc.n_newton == 0);
        CHECK(dsc.w_newton == 0.0);
    }
}

/**
 * @brief WHERE do the SME BoundaryCorner (d=0) nodes sit on the geodesic
 *        quarter hemisphere? — pinch-corner marginality dissection, step 1.
 *
 * The [sme_fail_dump] instances behind the [alpha_escalation] marginal
 * infeasibility each contain THREE adjacent zero-gap nodes, yet the
 * quarter-hemisphere model has only four GENUINE corners (two load
 * corners on the equator, two hole-rim/symmetry-plane corners). This
 * diag replicates the corner criterion of compute_sme_boundary_frames
 * (45° turn between incident boundary edges ⇒ BoundaryCorner ⇒ d=0,
 * paper §3.2.2) over the model's boundary, grouped by boundary curve
 * (hole rim / equator / symmetry planes), across resolutions.
 *
 * Question it answers: are the zero-gap clusters GENUINE corners of the
 * model, or spurious corner classifications of the JAGGED polygonal hole
 * rim (faces dropped by centroid ⇒ staircase polyline whose turn angles
 * exceed 45° regardless of how smooth the underlying hole is)?
 */
TEST_CASE("geodesic hemisphere: boundary-vertex corner classification "
          "by boundary curve",
          "[shell][sme][.diag][hemi_corner_cls]")
{
    constexpr double R = 10.0, hole_deg = 18.0;
    const double cos_hole = std::cos(hole_deg * std::numbers::pi / 180.0);
    constexpr double kCornerCos = 0.70710678;  // mirror of lme.cpp

    for (const int n : {16, 24, 32, 64}) {
        const auto mesh = build_quarter_hemisphere_geodesic(R, hole_deg, n);
        const auto bdry =
            chladni::shell::lme::collect_boundary_edges(mesh.F);
        const Eigen::Index n_v = mesh.V.rows();

        // Incident boundary-edge tangents per boundary vertex.
        std::vector<std::vector<Eigen::Vector3d>> tans(
            static_cast<std::size_t>(n_v));
        for (const auto& be : bdry) {
            const Eigen::Vector3d t =
                (mesh.V.row(be.v1) - mesh.V.row(be.v0))
                    .normalized()
                    .transpose();
            tans[static_cast<std::size_t>(be.v0)].push_back(t);
            tans[static_cast<std::size_t>(be.v1)].push_back(t);
        }

        // Boundary-curve membership (a vertex may sit on several).
        const double tol = 1.0e-6 * R;
        auto curve_of = [&](Eigen::Index v) -> const char* {
            const Eigen::Vector3d p = mesh.V.row(v).transpose();
            const bool on_sym = std::abs(p.x()) < tol
                             || std::abs(p.y()) < tol;
            const bool on_eq  = std::abs(p.z()) < tol;
            // Hole-rim nodes sit just below θ=hole_deg; the rim polyline
            // wanders ≲1 lattice step below the cut circle.
            const bool near_hole =
                p.z() / R > cos_hole - 2.5 / static_cast<double>(n);
            if (on_eq  && on_sym) return "corner:load";
            if (near_hole && on_sym) return "corner:hole-sym";
            if (on_eq)  return "equator";
            if (on_sym) return "symmetry";
            if (near_hole) return "hole-rim";
            return "other";
        };

        std::map<std::string, std::array<int, 2>> tally;  // {corner, mid}
        std::map<std::string, double> worst_cos;          // min |t_i·t_ref|
        for (Eigen::Index v = 0; v < n_v; ++v) {
            const auto& inc = tans[static_cast<std::size_t>(v)];
            if (inc.empty()) continue;
            double min_abs_cos = 1.0;
            for (std::size_t a = 0; a < inc.size(); ++a)
                for (std::size_t b = a + 1; b < inc.size(); ++b)
                    min_abs_cos = std::min(min_abs_cos,
                                           std::abs(inc[a].dot(inc[b])));
            const bool corner = min_abs_cos < kCornerCos;
            const std::string c = curve_of(v);
            tally[c][corner ? 0 : 1]++;
            auto it = worst_cos.find(c);
            if (it == worst_cos.end())
                worst_cos[c] = min_abs_cos;
            else
                it->second = std::min(it->second, min_abs_cos);
        }

        // Boundary-vertex TOTAL valence distribution (all incident
        // edges, not just boundary edges) — Loop's Schweitzer phantom
        // augmentation (augment_for_loop_boundary) only supports
        // boundary vertices of total valence 2, 3, or 4 (it closes the
        // fan to valence 6 with a fixed phantom count). The geodesic
        // octahedron-subdivision hole rim carries valence-5/6 boundary
        // vertices → Loop REJECTS the mesh. This is a connectivity
        // property of the triangulation, NOT the staircase positions:
        // the conformal rim-snap moves vertices but not edges, so it
        // would fail identically.
        {
            const auto edges = chladni::shell::build_edges(mesh.F.cast<int>());
            std::vector<int> val(static_cast<std::size_t>(n_v), 0);
            for (const auto& e : edges) {
                ++val[static_cast<std::size_t>(e.v0)];
                ++val[static_cast<std::size_t>(e.v1)];
            }
            std::vector<char> is_bdry(static_cast<std::size_t>(n_v), 0);
            for (const auto& be : bdry) {
                is_bdry[static_cast<std::size_t>(be.v0)] = 1;
                is_bdry[static_cast<std::size_t>(be.v1)] = 1;
            }
            std::map<int, int> vhist;       // boundary-vertex valence → count
            int n_unsupported = 0;
            for (Eigen::Index v = 0; v < n_v; ++v)
                if (is_bdry[static_cast<std::size_t>(v)]) {
                    const int vv = val[static_cast<std::size_t>(v)];
                    ++vhist[vv];
                    if (vv < 2 || vv > 4) ++n_unsupported;
                }
            std::printf("[hemi_corner_cls] n=%d boundary valence hist:", n);
            for (const auto& [vv, c] : vhist)
                std::printf(" v%d=%d", vv, c);
            std::printf("  | Loop-unsupported (valence∉{2,3,4}) = %d\n",
                        n_unsupported);
        }

        std::printf("\n[hemi_corner_cls] n=%d (%lld V, %zu bdry edges)\n",
                    n, static_cast<long long>(n_v), bdry.size());
        if (n == 16) {  // per-vertex detail at the failing resolution
            for (Eigen::Index v = 0; v < n_v; ++v) {
                const auto& inc = tans[static_cast<std::size_t>(v)];
                if (inc.empty()) continue;
                double min_abs_cos = 1.0;
                for (std::size_t a = 0; a < inc.size(); ++a)
                    for (std::size_t b = a + 1; b < inc.size(); ++b)
                        min_abs_cos = std::min(
                            min_abs_cos, std::abs(inc[a].dot(inc[b])));
                if (min_abs_cos >= kCornerCos) continue;  // corners only
                const Eigen::Vector3d p = mesh.V.row(v).transpose();
                std::string nbrs;
                for (const auto& be : bdry) {
                    if (be.v0 == v) nbrs += " " + std::to_string(be.v1);
                    if (be.v1 == v) nbrs += " " + std::to_string(be.v0);
                }
                std::printf("[hemi_corner_cls]   CORNER gid=%lld %s "
                            "p=(%.3f %.3f %.3f) theta=%.1f deg "
                            "turn=%.1f deg  rim-nbrs:%s\n",
                            static_cast<long long>(v), curve_of(v),
                            p.x(), p.y(), p.z(),
                            std::acos(p.z() / R) * 180.0
                                / std::numbers::pi,
                            std::acos(min_abs_cos) * 180.0
                                / std::numbers::pi,
                            nbrs.c_str());
            }
        }
        for (const auto& [c, t] : tally) {
            std::printf("[hemi_corner_cls]   %-16s corners=%3d mid=%3d "
                        "  worst turn = %5.1f deg\n",
                        c.c_str(), t[0], t[1],
                        std::acos(std::min(1.0, worst_cos[c])) * 180.0
                            / std::numbers::pi);
        }
    }
    SUCCEED("classification printed");
}

/**
 * @brief A/B: face-dropped vs CONFORMAL (rim-snapped) hole cut on the
 *        geodesic pinched hemisphere — fixture-faithfulness experiment
 *        motivated by the 2026-06-07 pinch-corner dissection.
 *
 * The shipped fixture drops faces by centroid, leaving a staircase
 * hole rim that (a) spuriously corner-classifies 4→12 rim nodes
 * (d=0 over-constraint along a smooth boundary, [hemi_corner_cls])
 * and (b) deviates from the paper's clean-rim discretizations. The
 * conformal variant projects rim vertices onto the exact θ=18° circle.
 *
 * Questions: do the spurious corners vanish; do the B3 pinch-corner
 * rescues persist (the GENUINE corner keeps d=0 either way); and does
 * SME α=2's flat δ/δref ≈ 1.10 plateau move toward Fig 17's ~1.0?
 */
TEST_CASE("geodesic hemisphere: conformal rim cut vs face-dropped — "
          "SME corners/rescues/deflection A/B",
          "[shell][sme][.diag][.slow][hemi_rim_snap]")
{
    constexpr double R = 10.0, t = 0.04, hole_deg = 18.0;
    constexpr double ref = 0.0924, F = 1.0;
    const auto mat = pinched_hemisphere_material();
    const chladni::shell::ShellMaterial sm =
        chladni::shell::shell_material_from_isotropic(mat, t);
    constexpr double kCornerCos = 0.70710678;  // mirror of lme.cpp

    auto count_corners = [&](const chladni::mesh::TriMesh& mesh) {
        const auto bdry =
            chladni::shell::lme::collect_boundary_edges(mesh.F);
        std::vector<std::vector<Eigen::Vector3d>> tans(
            static_cast<std::size_t>(mesh.V.rows()));
        for (const auto& be : bdry) {
            const Eigen::Vector3d tn =
                (mesh.V.row(be.v1) - mesh.V.row(be.v0))
                    .normalized()
                    .transpose();
            tans[static_cast<std::size_t>(be.v0)].push_back(tn);
            tans[static_cast<std::size_t>(be.v1)].push_back(tn);
        }
        int corners = 0;
        for (auto& inc : tans) {
            double mc = 1.0;
            for (std::size_t a = 0; a < inc.size(); ++a)
                for (std::size_t b = a + 1; b < inc.size(); ++b)
                    mc = std::min(mc, std::abs(inc[a].dot(inc[b])));
            if (!inc.empty() && mc < kCornerCos) ++corners;
        }
        return corners;
    };

    std::printf("\n=== Conformal-rim A/B (SME paper γ=0.8 α=2) ===\n");
    std::printf("%-4s %-12s %8s %8s %12s %10s %8s %8s\n",
                "n", "cut", "n_vert", "corners", "δ_r", "δ/δref",
                "rescues", "newton");

    for (const int n : {16, 24, 32, 48}) {
        for (const bool conform : {false, true}) {
            const auto mesh =
                build_quarter_hemisphere_geodesic(R, hole_deg, n, conform);
            const Eigen::Index n_v = mesh.V.rows();
            const double tol = 1.0e-6 * R;
            auto find = [&](const Eigen::Vector3d& p) {
                Eigen::Index best = 0; double bd = 1e30;
                for (Eigen::Index v = 0; v < n_v; ++v) {
                    const double d = (mesh.V.row(v).transpose() - p).norm();
                    if (d < bd) { bd = d; best = v; }
                }
                return best;
            };
            const Eigen::Index v_out = find({R, 0, 0});
            const Eigen::Index v_in  = find({0, R, 0});

            chladni::shell::LMEAssembler::Params p;
            p.use_second_order_sme = true;
            p.gamma     = 0.8;
            p.sme_alpha = 2.0;
            chladni::shell::LMEAssembler A{p};

            double dr = std::numeric_limits<double>::quiet_NaN();
            long rescues = -1, newton = -1;
            try {
                const Eigen::SparseMatrix<double> K =
                    A.assemble_K(mesh.V, mesh.F, sm);
                rescues = A.last_drop_stats().n_escal_ok;
                newton  = A.last_drop_stats().n_newton;
                std::vector<Eigen::Index> constrained;
                for (Eigen::Index v = 0; v < n_v; ++v) {
                    if (std::abs(mesh.V(v, 1)) < tol)
                        constrained.push_back(3 * v + 1);
                    if (std::abs(mesh.V(v, 0)) < tol)
                        constrained.push_back(3 * v + 0);
                }
                constrained.push_back(3 * v_out + 2);

                Eigen::VectorXd f = Eigen::VectorXd::Zero(K.rows());
                f(3 * v_out + 0) += F;
                f(3 * v_in  + 1) += -F;

                const Eigen::VectorXd U =
                    solve_static_dirichlet(K, f, constrained);
                dr = U(3 * v_out + 0);
                std::printf("%-4d %-12s %8lld %8d %12.6f %10.4f %8ld %8ld\n",
                            n, conform ? "conformal" : "face-drop",
                            static_cast<long long>(n_v),
                            count_corners(mesh), dr, dr / ref,
                            rescues, newton);
            } catch (const std::exception& e) {
                std::printf("%-4d %-12s %8lld %8d %12s %10s  (%s)\n",
                            n, conform ? "conformal" : "face-drop",
                            static_cast<long long>(n_v),
                            count_corners(mesh), "n/a", "-", e.what());
            }
            CHECK(std::isfinite(dr));  // both cuts must assemble+solve
        }
    }
    SUCCEED("A/B printed");
}

/**
 * @brief Pinched cylinder (Belytschko obstacle course) — CONFOUNDED.
 *
 * Kept for the record, but this turned out to be a poor discriminator:
 * the result mixes (a) the point-load representation on a non-interpolating
 * basis, (b) the coefficient→displacement read at an interior point, and
 * (c) the curved-shell bending under-stiffness isolated cleanly in the
 * flat-plate diagnostics above. Both Loop and LME OVERSHOOT and grow with
 * refinement (point-load deflection on a coarse mesh), so no clean
 * Loop-vs-LME convergence read is possible here. Use the flat-plate
 * diagnostics for the decisive evidence.
 */
TEST_CASE("pinched cylinder (Belytschko obstacle course): Loop vs LME "
          "normalised deflection under load (CONFOUNDED — see flat-plate)",
          "[shell][static][obstacle][pinched_cylinder][.diag]")
{
    // Increasing refinement; full-cylinder meshes (no symmetry reduction).
    // n_around even (θ=π load), n_along even (mid-length ring).
    const std::array<std::array<int, 2>, 3> levels = {{
        {{16, 8}}, {{24, 12}}, {{32, 16}},
    }};

    std::printf(
        "\n=== Pinched cylinder (L=600 R=300 t=3 E=3e6 nu=0.3, "
        "P=1, diaphragm ends) ===\n");
    std::printf("reference w_under_load = %.4e (normalised target = 1.0)\n",
                kPinchedCylRef);
    std::printf("%-8s %-8s %12s %12s %12s\n",
                "around", "along", "n_vert", "Loop w/wref", "LME w/wref");

    for (const auto& lv : levels) {
        const int na = lv[0];
        const int nl = lv[1];

        chladni::shell::LoopAssembler loop;
        const PinchedResult r_loop = run_pinched_cylinder(loop, na, nl);

        // Paper-faithful LME: curved shell + ghost nodes (defaults).
        chladni::shell::LMEAssembler lme{chladni::shell::LMEAssembler::Params{}};
        const PinchedResult r_lme = run_pinched_cylinder(lme, na, nl);

        std::printf("%-8d %-8d %12lld %12.4f %12.4f\n",
                    na, nl,
                    static_cast<long long>(r_loop.n_vertices),
                    r_loop.normalised, r_lme.normalised);

        // Sanity only: deflections must be finite and positive. The
        // quantitative read (does LME track Loop toward 1.0, or lock low?)
        // is the printed table — this test is diagnostic, not a hard gate.
        CHECK(std::isfinite(r_loop.normalised));
        CHECK(std::isfinite(r_lme.normalised));
        CHECK(r_loop.w_under_load > 0.0);
        CHECK(r_lme.w_under_load > 0.0);
    }
}
