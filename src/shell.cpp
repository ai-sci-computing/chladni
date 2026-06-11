/**
 * @file shell.cpp
 * @brief Thin-shell FEM building blocks: geometry, energy, stiffness, and lumped mass.
 *
 * The membrane energy uses a constant-strain-triangle (CST) plane-stress
 * FEM element; the bending energy uses the Wardetzky 2007 quadratic
 * curvature IBM (per-hinge cotangent-weighted rank-1 quadratic form).
 *
 * References:
 *  - @cite grinspun_2003_discrete_shells (geometric framework)
 *  - @cite wardetzky_2007_quadratic_curvature (IBM bending energy)
 */

#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/loop.hpp>

#include <Eigen/Geometry>  // for Vector3d::cross

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <Eigen/Eigenvalues>

#include <random>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chladni::shell {

Eigen::VectorXd face_areas(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F)
{
    Eigen::VectorXd A(F.rows());
    for (Eigen::Index i = 0; i < F.rows(); ++i) {
        // Convert each row to a 3-vector explicitly so .cross() resolves
        // to the dense vector cross product unambiguously.
        const Eigen::Vector3d v0 = V.row(F(i, 0)).head<3>();
        const Eigen::Vector3d v1 = V.row(F(i, 1)).head<3>();
        const Eigen::Vector3d v2 = V.row(F(i, 2)).head<3>();
        A(i) = 0.5 * (v1 - v0).cross(v2 - v0).norm();
    }
    return A;
}

namespace {

/// Reject vertex-non-manifold input (two surface sheets meeting at a single
/// point — e.g. two cones sharing an apex). Edge-manifoldness alone misses
/// this: every edge can still bound ≤2 faces while a vertex's incident faces
/// split into several disconnected fans. We test, per vertex, whether its
/// incident faces form a single edge-connected fan; >1 component is pinched.
void check_vertex_manifold(const Eigen::MatrixXi& F)
{
    // vertex -> incident (face-local-index, {link-vertex w0, w1}) corners.
    std::unordered_map<Eigen::Index,
                       std::vector<std::array<Eigen::Index, 2>>> link;
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index v  = F(f, k);
            const Eigen::Index w0 = F(f, (k + 1) % 3);
            const Eigen::Index w1 = F(f, (k + 2) % 3);
            link[v].push_back({w0, w1});
        }
    }

    for (const auto& [v, faces] : link) {
        const std::size_t m = faces.size();
        if (m < 2) continue;

        // Union-find over this vertex's incident faces; two faces are joined
        // when they share a link vertex w (i.e. the manifold edge (v, w)).
        std::vector<std::size_t> parent(m);
        std::iota(parent.begin(), parent.end(), std::size_t{0});
        auto find = [&](std::size_t i) {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        };
        std::unordered_map<Eigen::Index, std::size_t> first_seen;
        for (std::size_t i = 0; i < m; ++i) {
            for (const Eigen::Index w : faces[i]) {
                auto [it, inserted] = first_seen.try_emplace(w, i);
                if (!inserted) {
                    const std::size_t ra = find(it->second), rb = find(i);
                    if (ra != rb) parent[ra] = rb;
                }
            }
        }

        std::size_t components = 0;
        for (std::size_t i = 0; i < m; ++i) {
            if (find(i) == i) ++components;
        }
        if (components > 1) {
            throw std::runtime_error(
                "build_edges: non-manifold mesh — vertex "
                + std::to_string(static_cast<long long>(v)) + " has "
                + std::to_string(static_cast<long long>(components))
                + " disconnected face fans (surfaces pinched at one point)");
        }
    }
}

/// Reject meshes whose faces span more than one connected component.
/// @ref compute_shell_modes projects out a single 6-dimensional rigid-body
/// subspace V_rigid (sized to one global span). K disjoint surfaces carry
/// 6·K rigid-body modes; the extra 6·(K−1) are M-orthogonal to that single
/// span, so rigid_proj_sq ≈ 0 for them and they pass the rigid filter — the
/// solver then returns them as spurious physical ~0 Hz modes (only a
/// borderline-negative λ is otherwise caught). Surface the disconnected
/// input as an error rather than silently returning wrong frequencies.
/// Connectivity is over the vertices referenced by @p F, joined through
/// shared faces.
void check_single_component(const Eigen::MatrixXi& F)
{
    if (F.rows() == 0) return;

    const Eigen::Index        n_v = F.maxCoeff() + 1;
    std::vector<Eigen::Index> parent(static_cast<std::size_t>(n_v));
    std::iota(parent.begin(), parent.end(), Eigen::Index{0});
    auto find = [&](Eigen::Index i) {
        while (parent[static_cast<std::size_t>(i)] != i) {
            const Eigen::Index gp =
                parent[static_cast<std::size_t>(
                    parent[static_cast<std::size_t>(i)])];
            parent[static_cast<std::size_t>(i)] = gp;
            i                                   = gp;
        }
        return i;
    };
    auto unite = [&](Eigen::Index a, Eigen::Index b) {
        const Eigen::Index ra = find(a);
        const Eigen::Index rb = find(b);
        if (ra != rb) parent[static_cast<std::size_t>(ra)] = rb;
    };

    std::vector<char> referenced(static_cast<std::size_t>(n_v), 0);
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        const Eigen::Index v0 = F(f, 0);
        const Eigen::Index v1 = F(f, 1);
        const Eigen::Index v2 = F(f, 2);
        unite(v0, v1);
        unite(v0, v2);
        referenced[static_cast<std::size_t>(v0)] = 1;
        referenced[static_cast<std::size_t>(v1)] = 1;
        referenced[static_cast<std::size_t>(v2)] = 1;
    }

    Eigen::Index components = 0;
    for (Eigen::Index i = 0; i < n_v; ++i) {
        if (referenced[static_cast<std::size_t>(i)] && find(i) == i) {
            ++components;
        }
    }
    if (components > 1) {
        throw std::runtime_error(
            "build_edges: disconnected mesh — faces span "
            + std::to_string(static_cast<long long>(components))
            + " connected components. compute_shell_modes projects out a "
            "single 6-DOF rigid-body subspace, so a multi-component mesh "
            "returns spurious ~0 Hz modes; analyse one component at a time.");
    }
}

/// Reject face indices outside [0, V.rows()). A face index >= V.rows()
/// (malformed OBJ / face-vertex mismatch) writes out of bounds in
/// @ref lumped_vertex_masses and under-covers the rigid filter; a negative
/// index is equally invalid. @ref build_edges sizes its structures from @p F
/// alone (union-find from F.maxCoeff()+1) and never compares against
/// V.rows(), so the compute_shell_modes entry points are the one chokepoint
/// that sees both V and F.
void check_face_indices(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    if (F.rows() == 0) return;
    if (F.minCoeff() < 0 || F.maxCoeff() >= V.rows()) {
        throw std::runtime_error(
            "compute_shell_modes: face index out of range — F references a "
            "vertex outside [0, V.rows()="
            + std::to_string(static_cast<long long>(V.rows()))
            + "). Malformed mesh / face-vertex count mismatch.");
    }
}

}  // anonymous namespace

std::vector<Edge> build_edges(const Eigen::MatrixXi& F)
{
    check_vertex_manifold(F);
    check_single_component(F);

    using Key = std::pair<Eigen::Index, Eigen::Index>;
    std::map<Key, Edge> edges;

    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = F(f, k);
            const Eigen::Index b = F(f, (k + 1) % 3);
            const Key key{std::min(a, b), std::max(a, b)};
            auto [it, /*inserted*/ _] = edges.try_emplace(
                key, Edge{key.first, key.second, -1, -1});

            // The directed edge a->b is the "left" side iff a < b.
            if (a == key.first) {
                if (it->second.face_left != -1) {
                    throw std::runtime_error(
                        "build_edges: non-manifold mesh — edge "
                        + std::to_string(static_cast<long long>(a)) + "-"
                        + std::to_string(static_cast<long long>(b))
                        + " has two left-side faces");
                }
                it->second.face_left = f;
            } else {
                if (it->second.face_right != -1) {
                    throw std::runtime_error(
                        "build_edges: non-manifold mesh — edge "
                        + std::to_string(static_cast<long long>(a)) + "-"
                        + std::to_string(static_cast<long long>(b))
                        + " has two right-side faces");
                }
                it->second.face_right = f;
            }
        }
    }

    std::vector<Edge> out;
    out.reserve(edges.size());
    // std::map iterates in key (lexicographic v0, v1) order, so the
    // returned list is already sorted, matching the API contract.
    for (const auto& [/*key*/ _, e] : edges) {
        out.push_back(e);
    }
    return out;
}

Eigen::VectorXd lumped_vertex_masses(const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     double density,
                                     double thickness)
{
    if (density <= 0.0) {
        throw std::invalid_argument(
            "lumped_vertex_masses: density must be > 0");
    }
    if (thickness <= 0.0) {
        throw std::invalid_argument(
            "lumped_vertex_masses: thickness must be > 0");
    }

    const auto areas = face_areas(V, F);
    const double surface_density = density * thickness;

    Eigen::VectorXd m = Eigen::VectorXd::Zero(V.rows());
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        const double per_corner = surface_density * areas(f) / 3.0;
        for (int k = 0; k < 3; ++k) {
            m(F(f, k)) += per_corner;
        }
    }
    return m;
}

namespace {

/// Locate the "third" vertex of a triangle given the two other vertices.
/// Returns the vertex of @p face that is neither @p v_a nor @p v_b. The
/// triangle is assumed to contain both @p v_a and @p v_b.
Eigen::Index opposite_vertex(const Eigen::MatrixXi& F,
                             Eigen::Index face,
                             Eigen::Index v_a,
                             Eigen::Index v_b)
{
    for (int k = 0; k < 3; ++k) {
        const auto v = F(face, k);
        if (v != v_a && v != v_b) return v;
    }
    throw std::runtime_error(
        "opposite_vertex: face does not contain both endpoints");
}

}  // anonymous namespace

namespace {

/// Cotangent of the angle at vertex @p v formed by edges @p (v, a) and @p (v, b).
/// Implemented as cot(angle) = (e_a . e_b) / |e_a x e_b| with edge vectors
/// e_a = a - v, e_b = b - v. Throws on degenerate triangles.
double cot_angle(const Eigen::Vector3d& v,
                 const Eigen::Vector3d& a,
                 const Eigen::Vector3d& b)
{
    const Eigen::Vector3d ea = a - v;
    const Eigen::Vector3d eb = b - v;
    const double cross_norm = (ea.cross(eb)).norm();
    if (cross_norm <= 0.0) {
        throw std::runtime_error(
            "cot_angle: degenerate angle (collinear edges)");
    }
    return ea.dot(eb) / cross_norm;
}

}  // anonymous namespace

std::vector<EdgeRestData> compute_edge_rest_data(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges)
{
    std::vector<EdgeRestData> rd(edges.size());

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Edge& e = edges[i];
        EdgeRestData& d = rd[i];

        const Eigen::Vector3d p0 = V.row(e.v0).head<3>();
        const Eigen::Vector3d p1 = V.row(e.v1).head<3>();
        const Eigen::Vector3d edge = p1 - p0;
        d.length = edge.norm();
        if (d.length <= 0.0) {
            throw std::runtime_error(
                "compute_edge_rest_data: degenerate edge (zero length)");
        }
        d.hat = edge / d.length;

        // Wing vertices and dihedral are only defined on interior edges.
        d.c_left  = -1;
        d.c_right = -1;
        d.dihedral = 0.0;
        d.combined_area = 0.0;
        d.c_ibm.setZero();

        // Per-edge h: 1/3 the mean perpendicular height of the incident
        // triangles (Grinspun 2003 eq. (2)). Diagnostic only after the
        // IBM rewrite — kept for legacy callers / tests.
        auto height_to_edge = [&](Eigen::Index face) -> double {
            const Eigen::Vector3d a = V.row(F(face, 0)).head<3>();
            const Eigen::Vector3d b = V.row(F(face, 1)).head<3>();
            const Eigen::Vector3d c = V.row(F(face, 2)).head<3>();
            const double area = 0.5 * (b - a).cross(c - a).norm();
            return 2.0 * area / d.length;
        };

        double h_sum = 0.0;
        int    h_n   = 0;
        if (e.face_left  != -1) { h_sum += height_to_edge(e.face_left);  ++h_n; }
        if (e.face_right != -1) { h_sum += height_to_edge(e.face_right); ++h_n; }
        if (h_n == 0) {
            throw std::runtime_error(
                "compute_edge_rest_data: edge has no adjacent face");
        }
        d.h_e = (h_sum / static_cast<double>(h_n)) / 3.0;

        if (!e.is_interior()) continue;

        d.c_left  = opposite_vertex(F, e.face_left,  e.v0, e.v1);
        d.c_right = opposite_vertex(F, e.face_right, e.v0, e.v1);

        const Eigen::Vector3d cl = V.row(d.c_left).head<3>();
        const Eigen::Vector3d cr = V.row(d.c_right).head<3>();

        // Outward face normals (CCW winding → outward via right-hand rule).
        const Eigen::Vector3d n_l_raw = (p1 - p0).cross(cl - p0);
        const Eigen::Vector3d n_r_raw = (cr - p0).cross(p1 - p0);
        const double nl_norm = n_l_raw.norm();
        const double nr_norm = n_r_raw.norm();
        if (nl_norm <= 0.0 || nr_norm <= 0.0) {
            throw std::runtime_error(
                "compute_edge_rest_data: degenerate adjacent face (zero area)");
        }
        const Eigen::Vector3d n_l = n_l_raw / nl_norm;
        const Eigen::Vector3d n_r = n_r_raw / nr_norm;

        // Signed bend angle: rotation from n_L to n_R about the edge.
        // theta_e = atan2((n_L x n_R) . hat,  n_L . n_R)
        const double sin_t = (n_l.cross(n_r)).dot(d.hat);
        const double cos_t = n_l.dot(n_r);
        d.dihedral = std::atan2(sin_t, cos_t);

        // Combined rest area: 1/2 * |cross product| for each face.
        const double area_left  = 0.5 * nl_norm;
        const double area_right = 0.5 * nr_norm;
        d.combined_area = area_left + area_right;

        // Wardetzky IBM cotangent coefficients
        // (@cite wardetzky_2007_quadratic_curvature theorem 3 / eq. 8).
        // Paper labels: x_0, x_1 = hinge endpoints; x_2 = T_0 apex;
        // x_3 = T_1 apex. Map to chladni: x_0 ↔ v0, x_1 ↔ v1,
        // x_2 ↔ c_left, x_3 ↔ c_right; T_0 ↔ left, T_1 ↔ right.
        //
        // The angles α_0j are the interior angles at the hinge endpoints
        // (between e_0 and the other edge of the same triangle). The
        // four needed cotangents are:
        const double cot_v0_TL = cot_angle(p0, p1, cl);  // at v0 in T_left
        const double cot_v1_TL = cot_angle(p1, p0, cl);  // at v1 in T_left
        const double cot_v0_TR = cot_angle(p0, p1, cr);  // at v0 in T_right
        const double cot_v1_TR = cot_angle(p1, p0, cr);  // at v1 in T_right

        // Paper formulas:
        //   c_0 (= for x_0 = v0)     =  cot α_03 + cot α_04
        //                            =  cot(at v1 in T_L) + cot(at v1 in T_R)
        //   c_1 (= for x_1 = v1)     =  cot α_01 + cot α_02
        //                            =  cot(at v0 in T_L) + cot(at v0 in T_R)
        //   c_2 (= for x_2 = c_L)    = -cot α_01 - cot α_03
        //                            = -cot(at v0 in T_L) - cot(at v1 in T_L)
        //   c_3 (= for x_3 = c_R)    = -cot α_02 - cot α_04
        //                            = -cot(at v0 in T_R) - cot(at v1 in T_R)
        // Order in c_ibm matches the 12-vector concatenation
        // (v0, v1, c_left, c_right):
        d.c_ibm <<  (cot_v1_TL + cot_v1_TR),
                    (cot_v0_TL + cot_v0_TR),
                   -(cot_v0_TL + cot_v1_TL),
                   -(cot_v0_TR + cot_v1_TR);
        // Sanity: sum(c) = 0 (translation invariance).
    }

    return rd;
}

ShellMaterial shell_material_from_isotropic(
    const ::chladni::IsotropicMaterial& mat,
    double thickness)
{
    if (thickness <= 0.0) {
        throw std::invalid_argument(
            "shell_material_from_isotropic: thickness must be > 0");
    }
    if (mat.youngs_modulus <= 0.0) {
        throw std::invalid_argument(
            "shell_material_from_isotropic: Young's modulus must be > 0");
    }
    if (mat.poisson_ratio <= -1.0 || mat.poisson_ratio >= 0.5) {
        throw std::invalid_argument(
            "shell_material_from_isotropic: nu must lie in (-1, 1/2)");
    }
    const double E   = mat.youngs_modulus;
    const double nu  = mat.poisson_ratio;
    const double h   = thickness;
    const double one_minus_nu2 = 1.0 - nu * nu;

    return ShellMaterial{
        .k_L = E * h / one_minus_nu2,
        .k_B = E * h * h * h / (12.0 * one_minus_nu2),
        .poisson_ratio = nu,
    };
}

namespace {

/// Per-triangle CST rest data: the rest-frame in-plane tangents,
/// rest area, and the rest local 2D coordinates of the three corners.
/// Built from rest vertex positions only.
struct TriangleCstRest {
    Eigen::Vector3d t1;          ///< First in-plane tangent (rest local x).
    Eigen::Vector3d t2;          ///< Second in-plane tangent (rest local y).
    double          area;        ///< Rest triangle area.
    Eigen::Matrix<double, 3, 2> X_local;  ///< Rest 2D coords of the three vertices.
};

/// Build the CST rest data for a single triangle from its rest 3D positions.
/// Throws on degenerate triangles. Local 2D layout: vertex 0 at the origin,
/// vertex 1 along +x, vertex 2 in the upper half-plane.
TriangleCstRest compute_triangle_cst_rest(const Eigen::Vector3d& a,
                                          const Eigen::Vector3d& b,
                                          const Eigen::Vector3d& c)
{
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    const Eigen::Vector3d normal_raw = ab.cross(ac);
    const double normal_norm = normal_raw.norm();
    if (normal_norm <= 0.0) {
        throw std::runtime_error(
            "compute_triangle_cst_rest: degenerate triangle (zero area)");
    }

    TriangleCstRest out;
    out.area = 0.5 * normal_norm;

    const Eigen::Vector3d n = normal_raw / normal_norm;
    const double ab_norm = ab.norm();
    out.t1 = ab / ab_norm;
    out.t2 = n.cross(out.t1);  // already unit (n and t1 are unit & orthogonal)

    out.X_local(0, 0) = 0.0;
    out.X_local(0, 1) = 0.0;
    out.X_local(1, 0) = ab_norm;
    out.X_local(1, 1) = 0.0;
    out.X_local(2, 0) = ac.dot(out.t1);
    out.X_local(2, 1) = ac.dot(out.t2);
    return out;
}

/// Standard CST 2D strain-displacement matrix B (3x6) given the three
/// vertices' local 2D coordinates and the rest area. Rows of the
/// returned strain are (eps_xx, eps_yy, gamma_xy = 2 eps_xy) in Voigt
/// notation. Cook §3.2.
Eigen::Matrix<double, 3, 6> compute_cst_B(
    const Eigen::Matrix<double, 3, 2>& X_local, double area)
{
    const double inv_2A = 1.0 / (2.0 * area);
    const double x0 = X_local(0, 0), y0 = X_local(0, 1);
    const double x1 = X_local(1, 0), y1 = X_local(1, 1);
    const double x2 = X_local(2, 0), y2 = X_local(2, 1);

    const double b0 = y1 - y2, c0 = x2 - x1;
    const double b1 = y2 - y0, c1 = x0 - x2;
    const double b2 = y0 - y1, c2 = x1 - x0;

    Eigen::Matrix<double, 3, 6> B;
    B << b0,  0.0, b1,  0.0, b2,  0.0,
         0.0, c0,  0.0, c1,  0.0, c2,
         c0,  b0,  c1,  b1,  c2,  b2;
    B *= inv_2A;
    return B;
}

/// 3x3 plane-stress isotropic stiffness D (for D_struct in shell.hpp).
/// Caller supplies the prefactor @c k_L = E h / (1 - nu^2) and Poisson @c nu.
Eigen::Matrix3d plane_stress_D(double k_L, double nu)
{
    Eigen::Matrix3d D;
    D <<  1.0,        nu,   0.0,
          nu,         1.0,  0.0,
          0.0,        0.0,  0.5 * (1.0 - nu);
    D *= k_L;
    return D;
}

}  // anonymous namespace

double shell_energy(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXd& V_current,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material)
{
    if (V_rest.rows() != V_current.rows() || V_rest.cols() != V_current.cols()) {
        throw std::invalid_argument(
            "shell_energy: V_rest and V_current shape mismatch");
    }
    if (edges.size() != rest_data.size()) {
        throw std::invalid_argument(
            "shell_energy: rest_data size mismatch with edges");
    }

    double W = 0.0;

    // ---- Membrane (CST per triangle) ----------------------------------
    if (material.k_L != 0.0) {
        const Eigen::Matrix3d D = plane_stress_D(material.k_L, material.poisson_ratio);
        for (Eigen::Index f = 0; f < F.rows(); ++f) {
            const Eigen::Index ia = F(f, 0);
            const Eigen::Index ib = F(f, 1);
            const Eigen::Index ic = F(f, 2);
            const auto rest = compute_triangle_cst_rest(
                V_rest.row(ia).head<3>(),
                V_rest.row(ib).head<3>(),
                V_rest.row(ic).head<3>());
            const auto B = compute_cst_B(rest.X_local, rest.area);

            // Project per-vertex 3D displacement onto the rest local frame
            // and stack into the 6-vector u_local = (u0_x, u0_y, u1_x, u1_y, u2_x, u2_y).
            Eigen::Matrix<double, 6, 1> u_local;
            const std::array<Eigen::Index, 3> verts = {ia, ib, ic};
            for (int k = 0; k < 3; ++k) {
                const Eigen::Vector3d disp =
                    V_current.row(verts[k]).head<3>() - V_rest.row(verts[k]).head<3>();
                u_local(2 * k + 0) = disp.dot(rest.t1);
                u_local(2 * k + 1) = disp.dot(rest.t2);
            }

            // Strain at rest displacement is exactly zero, so this is the
            // linearised strain only — not Green-Lagrange. Sufficient for
            // the modal Hessian which only depends on the second derivative
            // at u = 0.
            const Eigen::Vector3d eps = B * u_local;
            W += 0.5 * rest.area * eps.dot(D * eps);
        }
    }

    // ---- Bending (Wardetzky IBM per interior edge) ---------------------
    if (material.k_B != 0.0) {
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const Edge& e = edges[i];
            // Boundary edges carry no IBM bending stencil (no c_right), so
            // they contribute zero bending stiffness. This is the intended
            // *free-edge* natural boundary condition, not an omission — see
            // the one-time boundary-edge count in assemble_stiffness_at_rest_fd
            // (SHELL_DIAG) if you need to confirm how many edges are free.
            if (!e.is_interior()) continue;
            const auto& rd = rest_data[i];

            // Stencil order: (v0, v1, c_left, c_right) — matches c_ibm.
            const std::array<Eigen::Index, 4> stencil =
                {e.v0, e.v1, rd.c_left, rd.c_right};
            Eigen::Vector3d q = Eigen::Vector3d::Zero();
            for (int k = 0; k < 4; ++k) {
                const Eigen::Vector3d disp =
                    V_current.row(stencil[k]).head<3>() - V_rest.row(stencil[k]).head<3>();
                q += rd.c_ibm(k) * disp;
            }
            // Energy: (3 * k_B / (2 * A_0)) * |q|^2.
            // 1/2 already absorbed into the prefactor, so this is
            // exactly the displacement-form Hessian (3 k_B / A_0) c c^T (x) I_3
            // times u, divided by 2.
            W += (1.5 * material.k_B / rd.combined_area) * q.squaredNorm();
        }
    }

    return W;
}

Eigen::SparseMatrix<double> assemble_stiffness_at_rest_fd(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material,
    double eps)
{
    if (eps <= 0.0) {
        throw std::invalid_argument(
            "assemble_stiffness_at_rest_fd: eps must be > 0");
    }
    if (edges.size() != rest_data.size()) {
        throw std::invalid_argument(
            "assemble_stiffness_at_rest_fd: rest_data size mismatch");
    }
    const Eigen::Index n   = V_rest.rows();
    const Eigen::Index dim = 3 * n;

    // One-time diagnostic: boundary edges carry no bending stiffness (the
    // free-edge BC). Surface how many there are so a "too soft" free plate is
    // not mistaken for a bug. Gated to keep the FD path quiet by default.
    if (std::getenv("SHELL_DIAG")) {
        std::size_t n_boundary = 0;
        for (const auto& e : edges) {
            if (!e.is_interior()) ++n_boundary;
        }
        std::cerr << "[SHELL_DIAG] assemble_stiffness_at_rest_fd: "
                  << n_boundary << " / " << edges.size()
                  << " edges are free (no bending stiffness)\n";
    }

    // Build the vertex coupling set from edge stencils. Vertex pair (a, b)
    // is "coupled" iff some edge's energy depends on both, i.e. they
    // appear together in some edge stencil. Self-pairs (a, a) are added
    // for the diagonal blocks.
    std::set<std::pair<Eigen::Index, Eigen::Index>> coupled;
    for (Eigen::Index v = 0; v < n; ++v) {
        coupled.emplace(v, v);
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        std::vector<Eigen::Index> stencil{edges[i].v0, edges[i].v1};
        if (edges[i].is_interior()) {
            stencil.push_back(rest_data[i].c_left);
            stencil.push_back(rest_data[i].c_right);
        }
        for (auto va : stencil) {
            for (auto vb : stencil) {
                coupled.emplace(va, vb);
            }
        }
    }

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> triplets;
    triplets.reserve(coupled.size() * 9);

    Eigen::MatrixXd V = V_rest;  // working copy; restored after each pair

    const double inv_eps2  = 1.0 / (eps * eps);
    const double inv_4eps2 = 0.25 * inv_eps2;

    // Energy at the linearisation point. The diagonal second difference is
    // (W(+e) - 2*W(0) + W(-e))/eps^2; with V == V_rest the rest energy W(0)
    // is identically zero, but we evaluate it explicitly so the formula stays
    // correct if a prestressed (V_current != V_rest) configuration is ever
    // passed in. Dropping the -2*W(0) term is the latent trap this guards.
    const double W0 = shell_energy(V_rest, V, F, edges, rest_data, material);

    for (const auto& [va, vb] : coupled) {
        for (Eigen::Index alpha = 0; alpha < 3; ++alpha) {
            for (Eigen::Index beta = 0; beta < 3; ++beta) {
                const Eigen::Index I = 3 * va + alpha;
                const Eigen::Index J = 3 * vb + beta;

                double k_ij;
                if (I == J) {
                    // Diagonal second difference: (W(+e) - 2*W(0) + W(-e))/eps^2.
                    V(va, alpha) += eps;
                    const double Wp = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) -= 2.0 * eps;
                    const double Wm = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) += eps;  // restore
                    k_ij = (Wp - 2.0 * W0 + Wm) * inv_eps2;
                } else {
                    // Off-diagonal central differences.
                    V(va, alpha) += eps;
                    V(vb, beta)  += eps;
                    const double Wpp = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) -= 2.0 * eps;
                    V(vb, beta)  -= 2.0 * eps;
                    const double Wmm = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) += 2.0 * eps;
                    const double Wpm = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) -= 2.0 * eps;
                    V(vb, beta)  += 2.0 * eps;
                    const double Wmp = shell_energy(V_rest, V, F, edges, rest_data, material);
                    V(va, alpha) += eps;       // restore va to original
                    V(vb, beta)  -= eps;       // restore vb to original
                    k_ij = (Wpp + Wmm - Wpm - Wmp) * inv_4eps2;
                }

                // Drop sub-noise entries to keep K's sparsity tight.
                if (std::abs(k_ij) > 1.0e-12) {
                    triplets.emplace_back(I, J, k_ij);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> K(dim, dim);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();

    // Numerical FD is symmetric in math but not exactly in floats.
    // Symmetrize to expose the property cleanly to downstream callers.
    Eigen::SparseMatrix<double> Kt = K.transpose();
    K = 0.5 * (K + Kt);
    K.makeCompressed();
    return K;
}

/// @brief Closed-form analytic stiffness assembly at rest.
///
/// Membrane: per-triangle CST 9x9 K_T = A_T * (B_2D P)^T D (B_2D P)
/// scattered into the global 3n×3n stiffness. Bending: per-hinge
/// IBM 12x12 K_e = (3 k_B / A_0) * c c^T (x) I_3 from
/// @cite wardetzky_2007_quadratic_curvature theorem 3, scattered into
/// the four-vertex stencil (v0, v1, c_left, c_right).
Eigen::SparseMatrix<double> assemble_stiffness_at_rest_analytic(
    const Eigen::MatrixXd& V_rest,
    const Eigen::MatrixXi& F,
    const std::vector<Edge>& edges,
    const std::vector<EdgeRestData>& rest_data,
    const ShellMaterial& material)
{
    if (edges.size() != rest_data.size()) {
        throw std::invalid_argument(
            "assemble_stiffness_at_rest_analytic: rest_data size mismatch");
    }
    const Eigen::Index n   = V_rest.rows();
    const Eigen::Index dim = 3 * n;

    std::vector<Eigen::Triplet<double>> triplets;
    // Heuristic: each triangle contributes 9x9 (81 triplets) for membrane,
    // each interior edge contributes 12x12 (144) for bending.
    triplets.reserve(static_cast<std::size_t>(81) * F.rows()
                   + static_cast<std::size_t>(144) * edges.size());

    /// Scatter a 3x3 sub-block @p block into rows starting at 3*va and
    /// columns starting at 3*vb.
    auto scatter_3x3 = [&](Eigen::Index va, Eigen::Index vb,
                           const Eigen::Matrix3d& block)
    {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double v = block(i, j);
                if (std::abs(v) > 1.0e-30) {
                    triplets.emplace_back(
                        static_cast<int>(3 * va + i),
                        static_cast<int>(3 * vb + j), v);
                }
            }
        }
    };

    // ---- Membrane (CST per triangle) -----------------------------------
    if (material.k_L != 0.0) {
        const Eigen::Matrix3d D_2D = plane_stress_D(material.k_L, material.poisson_ratio);
        for (Eigen::Index f = 0; f < F.rows(); ++f) {
            const std::array<Eigen::Index, 3> verts =
                {F(f, 0), F(f, 1), F(f, 2)};
            const auto rest = compute_triangle_cst_rest(
                V_rest.row(verts[0]).head<3>(),
                V_rest.row(verts[1]).head<3>(),
                V_rest.row(verts[2]).head<3>());
            const auto B_2D = compute_cst_B(rest.X_local, rest.area);

            // Projection matrix P (6x9): u_local_2D = P * u_3D where each
            // 2x3 block reads the dot product of the 3D displacement with
            // the in-plane tangents (t1, t2). B_3D = B_2D * P.
            Eigen::Matrix<double, 3, 9> B_3D;
            B_3D.setZero();
            for (int k = 0; k < 3; ++k) {
                B_3D.block<3, 3>(0, 3 * k) =
                    B_2D.block<3, 1>(0, 2 * k + 0) * rest.t1.transpose()
                  + B_2D.block<3, 1>(0, 2 * k + 1) * rest.t2.transpose();
            }
            const Eigen::Matrix<double, 9, 9> K_T =
                rest.area * (B_3D.transpose() * D_2D * B_3D);

            for (int bi = 0; bi < 3; ++bi) {
                for (int bj = 0; bj < 3; ++bj) {
                    const Eigen::Matrix3d block =
                        K_T.block<3, 3>(3 * bi, 3 * bj);
                    scatter_3x3(verts[bi], verts[bj], block);
                }
            }
        }
    }

    // ---- Bending (Wardetzky IBM per interior edge) ---------------------
    // This is the analytic Hessian path that compute_shell_modes actually
    // uses. As in the FD twin, boundary edges carry no IBM bending stencil
    // (no c_right) and so contribute zero bending stiffness — the intended
    // *free-edge* natural BC, not an omission. The one-time boundary-edge
    // count is emitted only by assemble_stiffness_at_rest_fd under
    // SHELL_DIAG; run that twin if you need to confirm how many edges are
    // free (the count is identical — same edge list).
    if (material.k_B != 0.0) {
        for (std::size_t k = 0; k < edges.size(); ++k) {
            const Edge& e = edges[k];
            if (!e.is_interior()) continue;
            const EdgeRestData& rd = rest_data[k];

            const std::array<Eigen::Index, 4> stencil =
                {e.v0, e.v1, rd.c_left, rd.c_right};
            const double coef = 3.0 * material.k_B / rd.combined_area;
            const Eigen::Vector4d& c = rd.c_ibm;

            // K_e = coef * c c^T (x) I_3. Off-diagonal block (p, q) is
            // coef * c_p * c_q * I_3; diagonal block is the same with p=q.
            for (int bi = 0; bi < 4; ++bi) {
                for (int bj = 0; bj < 4; ++bj) {
                    const double s = coef * c(bi) * c(bj);
                    Eigen::Matrix3d block = Eigen::Matrix3d::Zero();
                    block(0, 0) = s;
                    block(1, 1) = s;
                    block(2, 2) = s;
                    scatter_3x3(stencil[bi], stencil[bj], block);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> K(dim, dim);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    // Symmetric by construction (CST: B^T D B; IBM: c c^T (x) I_3 — both
    // exactly symmetric in math). Enforce numerically anyway.
    Eigen::SparseMatrix<double> Kt = K.transpose();
    K = 0.5 * (K + Kt);
    K.makeCompressed();
    return K;
}

namespace {

/// Solve the generalised eigenproblem @f$K \phi = \lambda M \phi@f$,
/// filter out rigid-body modes by mass-orthogonal projection against the
/// 6-dim translation+rotation subspace built from @p V, and return the
/// lowest @p n_modes physical modes ascending. Shared by
/// @ref chladni::shell::compute_shell_modes (CST + Wardetzky-IBM K) and
/// @ref chladni::shell::compute_shell_modes_loop (Cirak-Ortiz Loop K).
/// The two entry points only differ in how @p K is assembled.
ShellModes solve_modal_eigenproblem_with_rigid_filter(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    const Eigen::MatrixXd&              V,
    std::size_t                          n_modes,
    const char*                          source_name)
{
    // Eigensolve K phi = lambda M phi via multi-seed Spectra shift-invert
    // with Rayleigh-Ritz fusion.
    //
    // ---------------------------------------------------------------------
    // Why multi-seed? Why not just one Spectra call, or PRIMME, or FEAST?
    // ---------------------------------------------------------------------
    // Spectra (and ARPACK, which Spectra re-implements) uses Implicitly
    // Restarted Arnoldi: Lanczos on the shift-invert operator
    // T = (K - sigma M)^-1 M, with periodic restart. The iteration starts
    // from a fixed-seed (seed 0) random initial residual.
    //
    // On meshes with *exact* discrete symmetry — most notably the
    // icosahedrally symmetric closed icosphere — the n=2 spheroidal cluster
    // is *exactly* 5-fold degenerate (Schur's lemma on the ℓ=2 irrep of the
    // icosahedral group). Lanczos applied to a 5-fold degenerate cluster
    // can lose orthogonality and converge to only 4 of the 5 invariant
    // directions, replacing the 5th with a "Krylov ghost" — a Ritz vector
    // outside the true invariant subspace with a nearby Ritz value. The
    // measured symptom on icosphere k=5 with the Stam K assembly was
    // 4 modes at 5905.5 Hz + 1 ghost at 5617 Hz (~5% spurious split).
    //
    // The bug is real and well-isolated. Probes pinned the cause to
    // starting-vector blindness, not K/M assembly: the seed-0 initial
    // residual happens to have anomalously small projection on one
    // specific direction of the 5-D invariant subspace, and Lanczos can
    // never recover what it doesn't sample. Different seeds miss
    // *different* directions — seed 0x3039 misses a different one. See
    // memory: chladni-spectra-degenerate-cluster.
    //
    // Why not PRIMME (modern block JD + LOBPCG)?
    //   PRIMME would categorically fix this — its block iteration with
    //   Rayleigh-Ritz at every step handles degenerate clusters by
    //   construction. But PRIMME's build is non-trivial: their .c files
    //   are templated via heavy macros, and the .h files are generated
    //   at build time by a Python script (tools/ctemplate). Integrating
    //   PRIMME via CMake means either replicating their Python generator
    //   in CMake, or using ExternalProject_Add over their makefile —
    //   either way adding python3 and LAPACK to the build prereqs and
    //   ~full day of CMake work. The chladni build promise is
    //   "fresh checkout requires only CMake + C++ compiler" (root
    //   CMakeLists.txt comment); PRIMME breaks that.
    //
    // Why not FEAST (contour-integration eigensolver)?
    //   FEAST slices the spectrum with contour integration, so cluster
    //   correctness is structural rather than best-effort. Pure Fortran 95,
    //   MIT licensed, simpler build than PRIMME (no python). But it adds
    //   gfortran to the build prereqs, which similarly breaks the
    //   "CMake + C++ compiler only" promise.
    //
    // Why multi-seed + Rayleigh-Ritz fusion works:
    //   Each seed run captures a different 4 of the 5 pentet directions.
    //   The union of 3 seed runs spans the full 5-D invariant subspace
    //   plus generous buffer in the rest of the spectrum. Rayleigh-Ritz
    //   on the combined subspace then gives the BEST polynomial
    //   approximation to the true eigenvalues within that span — and
    //   when the span CONTAINS the true invariant subspace, those
    //   approximations are exact to floating-point precision. Measured
    //   pentet rel_spread on icosphere k=5 Stam: 2.77e-15 (machine
    //   epsilon). Measured runtime cost: 3.2x single-Spectra (3
    //   factorisations of K - sigma M dominate).
    //
    // The cost trade is: 3.2x slower eigensolve for bit-exact
    // resolution of degenerate clusters on perfectly symmetric meshes.
    // For typical meshes (no exact symmetry — bunny, ellipsoid, real
    // scanned objects) the clusters are slightly split anyway and a
    // single Spectra run is also correct; we still pay the 3.2x cost
    // there, but the eigensolve is rarely the user-facing bottleneck.
    // See tests/shell/test_loop_sphere_projected.cpp for the timing
    // probes (probes 9-14) and the bit-exact heal verification.
    //
    // Memory note: the eigensolve here is cheap — K, M and the shift-invert
    // factorisation are all small (K at icosphere k=5 is ~140 MB sparse, its
    // factor far smaller; the Rayleigh-Ritz blocks are dim x O(nev)). An
    // earlier comment claimed a ~18 GB "OOM ceiling" intrinsic to factorising
    // the wide meshfree stencil, fixable only by a factorisation-free solver.
    // That was WRONG — MEASURED 2026-06-11: the wall was the curved ASSEMBLY
    // emitting one triplet block per (a,b) per *triangle* in their shared
    // support (a ~30-70x over-emission that only merged at setFromTriplets),
    // peaking at ~6 GB of triplets for K alone. Folding each (a,b) across the
    // whole thread before emitting (see assemble_K_curved_bending /
    // assemble_M_curved) cut icosphere k=5 SME from OOM(>18 GB) to ~3 GB and
    // unblocked that refinement level entirely. The solver was never the cap.
    // ---------------------------------------------------------------------

    // Slack absorbs the 6 rigid-body modes plus a margin for cases
    // where the shift-invert iteration loses a few eigenvalues to
    // ill-conditioning before reporting "Successful". On thin shells
    // the membrane / bending stiffness ratio scales as 12/h^2, so K
    // routinely has eigenvalues spanning ~10 orders of magnitude — the
    // lowest physical modes can be silently dropped from the lowest-N
    // returned set if the Krylov subspace is too tight (the symptom
    // user-side: dropping h to ~0.5 mm makes the audible bending modes
    // disappear from the spectrum even though they exist).
    constexpr Eigen::Index kRigidSlack = 16;
    const Eigen::Index dim = K.rows();
    const Eigen::Index nev =
        static_cast<Eigen::Index>(n_modes) + kRigidSlack;
    if (static_cast<Eigen::Index>(n_modes) >= dim) {
        throw std::invalid_argument(
            std::string(source_name)
            + ": n_modes exceeds the number of degrees of freedom available");
    }

    // Eigenpairs feeding the rigid-body filter below: evals ascending,
    // evecs with M-orthonormal columns. Produced either by a direct dense
    // generalised solve (tiny meshes) or the multi-seed shift-invert +
    // Rayleigh-Ritz fusion (everything else).
    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;

    // For tiny systems the requested window (nev) approaches the full
    // dimension — exactly the regime where shift-invert Lanczos is least
    // reliable: the Krylov subspace is nearly the whole space, restart
    // logic degenerates, and the σ = -1 shift can sit pathologically close
    // to a true eigenvalue. Solve these densely and exactly instead. Cost
    // is O(dim^3) but dim is small here by construction.
    constexpr Eigen::Index kDenseDimThreshold = 300;
    const bool use_dense = (dim <= kDenseDimThreshold) || (nev >= dim);

    if (use_dense) {
        const Eigen::MatrixXd Kd(K);
        const Eigen::MatrixXd Md(M);
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
            Kd, Md, Eigen::ComputeEigenvectors);
        if (ges.info() != Eigen::Success) {
            throw std::runtime_error(
                std::string(source_name)
                + ": dense generalised eigensolve (tiny-mesh fallback) failed");
        }
        // Eigen returns ascending eigenvalues with M-orthonormal eigenvectors,
        // matching the contract the rigid filter below expects.
        evals = ges.eigenvalues();
        evecs = ges.eigenvectors();
    } else {
        using OpKM = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
        using OpM  = Spectra::SparseSymMatProd<double>;
        OpKM op_km(K, M);
        OpM  op_m(M);

        // Generous Krylov subspace — Spectra's documentation recommends
        // ncv >= 2*nev+1, but for ill-conditioned PSD problems with widely
        // separated eigenvalue clusters (typical thin-shell K) 4*nev gives
        // dramatically more reliable convergence to the lowest cluster.
        const Eigen::Index ncv =
            std::min<Eigen::Index>(
                std::max<Eigen::Index>(4 * nev + 1, 60), dim);

        constexpr double kSigma = -1.0;  // small negative shift
        constexpr int    kMaxIters = 1000;
        constexpr double kSolverTol = 1.0e-10;

        // Three seed runs. The choice of three (vs two or four) is empirical:
        // probes show two runs sometimes miss the same direction by chance,
        // four is wasted work — three reliably captures every invariant
        // direction of the n=2 pentet on icosphere k=5 Stam (the worst-case
        // fixture in the test suite). Seeds are hard-coded so the eigensolve
        // is deterministic across runs.
        const std::array<std::uint32_t, 3> kSeeds = {
            0u, 1u, 0x3039u};
        std::vector<Eigen::MatrixXd> V_runs;
        V_runs.reserve(kSeeds.size());
        for (std::uint32_t seed : kSeeds) {
            Spectra::SymGEigsShiftSolver<OpKM, OpM, Spectra::GEigsMode::ShiftInvert>
                solver(op_km, op_m, nev, ncv, kSigma);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(-0.5, 0.5);
            Eigen::VectorXd init_resid(dim);
            for (Eigen::Index i = 0; i < dim; ++i) {
                init_resid(i) = uni(rng);
            }
            solver.init(init_resid.data());
            solver.compute(Spectra::SortRule::LargestMagn, kMaxIters, kSolverTol);
            if (solver.info() != Spectra::CompInfo::Successful) {
                throw std::runtime_error(
                    std::string(source_name)
                    + ": Spectra shift-invert eigensolve did not converge "
                      "(multi-seed run failed)");
            }
            V_runs.push_back(solver.eigenvectors());
        }

        // Concatenate the three captured subspaces into a (3 * nev)-wide
        // candidate basis, then M-orthonormalise via modified Gram-Schmidt
        // in the M-inner-product. Columns whose M-norm² falls below 1e-20
        // after deflation against the running basis are dropped — these
        // are directions already spanned by earlier columns, expected
        // because the three seed runs each capture a 4-of-5 subset of the
        // n=2 pentet (so ~3x redundancy on that cluster) plus essentially
        // the same higher modes.
        Eigen::Index total_cols = 0;
        for (const auto& V_run : V_runs) total_cols += V_run.cols();
        Eigen::MatrixXd Combined(dim, total_cols);
        {
            Eigen::Index off = 0;
            for (const auto& V_run : V_runs) {
                Combined.middleCols(off, V_run.cols()) = V_run;
                off += V_run.cols();
            }
        }

        Eigen::MatrixXd Q(dim, Combined.cols());
        Eigen::Index q_cols = 0;
        for (Eigen::Index j = 0; j < Combined.cols(); ++j) {
            Eigen::VectorXd v = Combined.col(j);
            // SymGEigsShiftSolver returns M-orthonormal eigenvectors, so a
            // single MGS pass is sufficient; the second pass is cheap
            // insurance against numerical drift accumulating across the
            // three concatenated blocks.
            for (int pass = 0; pass < 2; ++pass) {
                for (Eigen::Index kk = 0; kk < q_cols; ++kk) {
                    const double c = Q.col(kk).dot(M * v);
                    v -= c * Q.col(kk);
                }
            }
            const double m_norm_sq = v.dot(M * v);
            if (m_norm_sq > 1.0e-20) {
                Q.col(q_cols) = v / std::sqrt(m_norm_sq);
                ++q_cols;
            }
        }
        Q.conservativeResize(Eigen::NoChange, q_cols);

        // Rayleigh-Ritz: project K and M onto the M-orthonormal subspace
        // Q, then solve the small dense generalised eigenproblem
        //   K' y = mu M' y,   K' = Q^T K Q,   M' = Q^T M Q (≈ I).
        // Eigen's GeneralizedSelfAdjointEigenSolver returns the mu in
        // ascending order, which directly gives our lambdas in ascending
        // order (so the rigid-body residues sit at the front for the
        // mass-projection filter below).
        const Eigen::MatrixXd KQ = K * Q;
        const Eigen::MatrixXd MQ = M * Q;
        Eigen::MatrixXd K_proj_raw = Q.transpose() * KQ;
        Eigen::MatrixXd M_proj_raw = Q.transpose() * MQ;
        const Eigen::MatrixXd K_proj =
            0.5 * (K_proj_raw + K_proj_raw.transpose());
        const Eigen::MatrixXd M_proj =
            0.5 * (M_proj_raw + M_proj_raw.transpose());

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(
            K_proj, M_proj, Eigen::ComputeEigenvectors);
        if (ges.info() != Eigen::Success) {
            throw std::runtime_error(
                std::string(source_name)
                + ": Rayleigh-Ritz refinement (dense generalised eigensolve "
                  "on the multi-seed subspace) failed");
        }
        evals = ges.eigenvalues();
        evecs = Q * ges.eigenvectors();
    }

    // Filter rigid-body modes via explicit mass-orthogonal projection
    // against the 6-dimensional rigid-body subspace V_rigid spanned by
    // the three translations and three rotations of the mesh as a
    // whole. For any mass-orthonormal eigenvector @f$\phi_i@f$ from
    // Spectra, the squared M-norm of its rigid-subspace projection is
    //   @f$ \|P_R \phi_i\|_M^2 \;=\; c_i^T \, G^{-1} \, c_i, @f$
    // where @f$ c_i = V_R^T M \phi_i @f$ and
    //  @f$ G = V_R^T M V_R @f$ is the 6x6 rigid Gram matrix in the
    // mass inner product. Because @f$\phi_i@f$ is M-normalised this
    // projection is bounded in [0, 1]: it is 1 when @f$\phi_i@f$ lies
    // fully in the rigid subspace and 0 when it is M-orthogonal to it.
    // A threshold of 0.5 cleanly separates the two — physical bending
    // modes have projection ~1e-6, rigid residues ~1.0 (verified on
    // cylinder, bunny, and ellipsoid).
    //
    // R8 (review 2026-06-09) questioned whether 0.5 could mis-classify a
    // borderline 40–60 %-rigid mode. It cannot: K's exact null space IS the
    // 6-dim rigid subspace, so for the symmetric (K, M) pencil every physical
    // eigenvector — a DISTINCT eigenvalue — is M-orthogonal to the rigid
    // kernel and has identically-zero projection. The split is therefore
    // structural, not tuned. Measured rigid_proj_sq is machine-bimodal (rigid
    // = 1.0, physical <= ~1e-22, margin ~1.0) across closed icospheres and
    // the free-free cylinder; see tests/shell/test_rigid_filter_distribution
    // .cpp ([rigid_dist]), the standing regression guard for this invariant.
    // An adaptive gap rule would be a no-op, so the hard 0.5 threshold stays.
    //
    // The previous spectrum-gap heuristic was insufficient: the
    // discrete-shells edge-spring K has a small but non-zero residual
    // for rigid rotations (~1e-7 relative; floating-point accumulation
    // in the cot/cosα formulae of the analytic Hessian) which on a
    // free-free cylinder produced a spurious mode whose frequency
    // grows linearly with axial element count and lands MID-spectrum
    // — between rigid-kernel residues and real physical modes — so
    // no gap can detect it. Mass-projection makes the classification
    // unambiguous regardless of where the spurious frequencies fall.
    //
    // Reference shell-FEM context: deflating the rigid-body subspace
    // before/after the eigensolve is the standard treatment for
    // free-free structural modal problems (Bathe "Finite Element
    // Procedures" §11.6 "Subspace iteration with deflation").
    // V_rigid is sized to K.rows() rather than 3*V.rows() so the rigid-
    // body subspace covers the FULL DOF set the eigenproblem was solved
    // on. For LME with ghost nodes (Millán 2011 §4.1.2) the assembled
    // system is 3*(N+G) > 3*N; ghost rows of V_rigid are left zero (we
    // don't have ghost positions here, and the real-vertex rows alone
    // capture > 50% of any whole-mesh rigid motion's M-norm² for typical
    // G << N, which is enough to clear the 0.5 classification threshold
    // below). For non-ghost paths K.rows() == 3*V.rows() so this is
    // byte-identical to the previous N-sized V_rigid.
    Eigen::MatrixXd V_rigid =
        Eigen::MatrixXd::Zero(K.rows(), 6);
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        const double x = V(i, 0);
        const double y = V(i, 1);
        const double z = V(i, 2);
        // Three translations.
        V_rigid(3 * i + 0, 0) = 1.0;
        V_rigid(3 * i + 1, 1) = 1.0;
        V_rigid(3 * i + 2, 2) = 1.0;
        // Three rotations about the global axes:
        //   ω × x at vertex (x, y, z) for ω = ê_x, ê_y, ê_z.
        V_rigid(3 * i + 1, 3) = -z;  // about x: (0, -z,  y)
        V_rigid(3 * i + 2, 3) =  y;
        V_rigid(3 * i + 0, 4) =  z;  // about y: (z, 0, -x)
        V_rigid(3 * i + 2, 4) = -x;
        V_rigid(3 * i + 0, 5) = -y;  // about z: (-y, x, 0)
        V_rigid(3 * i + 1, 5) =  x;
    }
    const Eigen::MatrixXd MV_rigid = M * V_rigid;
    const Eigen::Matrix<double, 6, 6> G_rigid =
        V_rigid.transpose() * MV_rigid;
    const Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol_G_rigid(G_rigid);
    if (chol_G_rigid.info() != Eigen::Success) {
        throw std::runtime_error(
            std::string(source_name)
            + ": rigid-body Gram matrix is not positive definite — the six "
              "rigid modes are linearly dependent (degenerate collinear "
              "geometry), so the rigid-body projection filter cannot run "
              "reliably.");
    }

    constexpr double kRigidProjThreshold = 0.5;
    std::vector<std::pair<double, Eigen::Index>> physical;
    physical.reserve(static_cast<std::size_t>(evals.size()));
    for (Eigen::Index i = 0; i < evals.size(); ++i) {
        const Eigen::Matrix<double, 6, 1> c =
            MV_rigid.transpose() * evecs.col(i);
        const Eigen::Matrix<double, 6, 1> g_inv_c = chol_G_rigid.solve(c);
        const double rigid_proj_sq = c.dot(g_inv_c);
        if (rigid_proj_sq > kRigidProjThreshold) continue;
        // Reject numerically non-positive eigenvalues. K is PSD and M is
        // SPD so λ ≥ 0 mathematically — but Spectra shift-invert near
        // σ = -1 can return a borderline rigid mode with λ slightly
        // negative (the mass-projection filter above catches MOST such
        // modes but the 0.5 threshold is a sharp cutoff and the rigid
        // cluster is exactly on the boundary). If we let λ < 0 through,
        // out.omegas = sqrt(λ) produces NaN downstream, which propagates
        // to the audio bank and the viz (user reported "mesh disappears"
        // on closed meshes at thin h where the borderline mode escapes
        // the projection cutoff). A non-positive λ that wasn't already
        // caught by the projection filter is treated as another
        // numerical rigid residue and dropped.
        if (!(evals(i) > 0.0)) continue;
        physical.emplace_back(evals(i), i);
    }
    if (physical.size() < n_modes) {
        throw std::runtime_error(
            std::string(source_name)
            + ": too many rigid-body / spurious near-zero modes in the "
              "leading window (kRigidSlack=" + std::to_string(kRigidSlack)
            + ", a compile-time constant — reaching this indicates a "
              "degenerate mesh or a non-PSD stiffness matrix, not a tunable "
              "parameter).");
    }
    // Sort ascending by eigenvalue regardless of Spectra's internal ordering.
    std::sort(physical.begin(), physical.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    ShellModes out;
    out.omegas.resize(static_cast<Eigen::Index>(n_modes));
    out.shapes.resize(K.rows(), static_cast<Eigen::Index>(n_modes));
    for (std::size_t k = 0; k < n_modes; ++k) {
        const Eigen::Index col = physical[k].second;
        out.omegas(static_cast<Eigen::Index>(k)) = std::sqrt(physical[k].first);
        out.shapes.col(static_cast<Eigen::Index>(k)) = evecs.col(col);
    }
    return out;
}

}  // namespace

ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    double thickness,
    std::size_t n_modes)
{
    return compute_shell_modes(
        V, F, material,
        shell_material_from_isotropic(material, thickness),
        thickness, n_modes);
}

ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double thickness,
    std::size_t n_modes)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "compute_shell_modes: n_modes must be >= 1");
    }
    if (thickness <= 0.0) {
        throw std::invalid_argument(
            "compute_shell_modes: thickness must be > 0");
    }

    check_face_indices(V, F);
    const auto edges = build_edges(F);
    const auto rd    = compute_edge_rest_data(V, F, edges);

    const auto vmasses = lumped_vertex_masses(V, F, material.density, thickness);
    auto M = assemble_mass_matrix(vmasses);

    auto K = assemble_stiffness_at_rest_analytic(V, F, edges, rd, sm);

    return solve_modal_eigenproblem_with_rigid_filter(
        K, M, V, n_modes, "compute_shell_modes");
}

ShellModes compute_shell_modes_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    double      thickness,
    std::size_t n_modes,
    int         n_passes,
    bool        use_stam)
{
    return compute_shell_modes_loop(
        V, F, material,
        shell_material_from_isotropic(material, thickness),
        thickness, n_modes, n_passes, use_stam);
}

ShellModes compute_shell_modes_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double      thickness,
    std::size_t n_modes,
    int         n_passes,
    bool        use_stam)
{
    // Back-compat shim: build a LoopAssembler with shipped defaults
    // (7-pt Dunavant K/M, consistent Galerkin mass / MassLumping::None
    // since 2026-05-17 late — flipped from RowSum after the lumping
    // bias was diagnosed) plus the caller's n_passes / use_stam, and
    // delegate.
    LoopAssembler::Params p;
    p.n_passes = n_passes;
    p.use_stam = use_stam;
    return compute_shell_modes(
        V, F, material, sm, thickness, n_modes, LoopAssembler{p});
}

ShellModes compute_shell_modes(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ::chladni::IsotropicMaterial& material,
    const ShellMaterial& sm,
    double      thickness,
    std::size_t n_modes,
    const ShellAssembler& assembler)
{
    if (n_modes == 0) {
        throw std::invalid_argument(
            "compute_shell_modes: n_modes must be >= 1");
    }
    if (thickness <= 0.0) {
        throw std::invalid_argument(
            "compute_shell_modes: thickness must be > 0");
    }

    // Validate mesh topology uniformly for every assembler path. The LME
    // assembler only reaches build_edges (and thus these checks) via
    // collect_boundary_edges, which is gated on use_ghost_nodes /
    // use_second_order_sme; with both off a disconnected or out-of-range
    // mesh would otherwise skip every guard and return spurious ~0 Hz modes.
    check_face_indices(V, F);
    check_single_component(F);

    const double surface_density = material.density * thickness;
    auto K = assembler.assemble_K(V, F, sm);
    auto M = assembler.assemble_M(V, F, surface_density);

    auto modes = solve_modal_eigenproblem_with_rigid_filter(
        K, M, V, n_modes, "compute_shell_modes");

    // Project the assembled-basis eigenvectors into vertex-displacement
    // coordinates so downstream consumers (viz, audio bank, mode-gallery)
    // see displacements regardless of the assembler's basis. Interpolating
    // bases (Loop, CST) return the input unchanged; meshfree bases (LME)
    // apply the sparse basis-evaluation matrix T_{ji} = N_i(x_j).
    modes.shapes = assembler.evaluate_modes_at_vertices(V, F, modes.shapes);
    return modes;
}

Eigen::SparseMatrix<double>
assemble_mass_matrix(const Eigen::VectorXd& vertex_masses)
{
    const Eigen::Index n   = vertex_masses.size();
    const Eigen::Index dim = 3 * n;

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> triplets;
    triplets.reserve(static_cast<std::size_t>(dim));
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index k = 0; k < 3; ++k) {
            const Eigen::Index d = 3 * i + k;
            triplets.emplace_back(d, d, vertex_masses(i));
        }
    }
    Eigen::SparseMatrix<double> M(dim, dim);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();
    return M;
}

}  // namespace chladni::shell
