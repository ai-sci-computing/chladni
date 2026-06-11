/**
 * @file loop.cpp
 * @brief Implementation of the Loop subdivision patch-stencil enumerator.
 *
 * See @ref include/chladni/shell/loop.hpp for the algorithm overview
 * and citations to @cite cirak_ortiz_schroder_2000_subdivision_shells
 * and @cite stam_1999_loop_evaluation.
 */

#include <chladni/shell/loop.hpp>

#include <Eigen/Geometry>  // Vector3d::cross
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace chladni::shell::loop {

std::vector<int> vertex_valences(
    Eigen::Index n_vertices,
    const std::vector<Edge>& edges)
{
    std::vector<int> valences(static_cast<std::size_t>(n_vertices), 0);
    for (const auto& e : edges) {
        ++valences[static_cast<std::size_t>(e.v0)];
        ++valences[static_cast<std::size_t>(e.v1)];
    }
    return valences;
}

namespace {

/**
 * @brief Map vertex index to the list of incident face indices.
 *
 * Built once per call; each face index appears exactly 3 times across
 * the lists (once per vertex). Memory is O(3 * |F|).
 */
std::vector<std::vector<Eigen::Index>>
build_vertex_to_faces(Eigen::Index n_vertices, const Eigen::MatrixXi& F)
{
    std::vector<std::vector<Eigen::Index>> v2f(
        static_cast<std::size_t>(n_vertices));
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            v2f[static_cast<std::size_t>(F(f, k))].push_back(f);
        }
    }
    return v2f;
}

/**
 * @brief Mark vertices that are an endpoint of any boundary edge.
 *
 * @return Length-@p n_vertices boolean vector with @c true at every
 *         boundary-vertex index.
 */
std::vector<bool>
mark_boundary_vertices(Eigen::Index n_vertices,
                       const std::vector<Edge>& edges)
{
    std::vector<bool> is_bdry(static_cast<std::size_t>(n_vertices), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }
    return is_bdry;
}

}  // namespace

std::vector<PatchStencil> build_patch_stencils(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    const Eigen::Index n_vertices = V.rows();
    const auto edges    = build_edges(F);
    const auto valences = vertex_valences(n_vertices, edges);
    const auto v2f      = build_vertex_to_faces(n_vertices, F);
    const auto is_bdry  = mark_boundary_vertices(n_vertices, edges);

    std::vector<PatchStencil> patches;
    patches.reserve(static_cast<std::size_t>(F.rows()));

    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        PatchStencil p;
        p.tri_index = f;
        for (int k = 0; k < 3; ++k) {
            p.corners[static_cast<std::size_t>(k)] = F(f, k);
            p.corner_valences[static_cast<std::size_t>(k)] =
                valences[static_cast<std::size_t>(F(f, k))];
        }

        // Build the ring as the union of the 1-rings of all 3 corners,
        // excluding the corners themselves. The 1-ring of a corner is
        // the set of vertices shared by faces incident to the corner.
        std::unordered_set<Eigen::Index> ring_set;
        bool any_boundary = false;
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index c = F(f, k);
            if (is_bdry[static_cast<std::size_t>(c)]) any_boundary = true;
            for (Eigen::Index g : v2f[static_cast<std::size_t>(c)]) {
                if (g == f) continue;
                for (int j = 0; j < 3; ++j) {
                    const Eigen::Index w = F(g, j);
                    if (w == p.corners[0] || w == p.corners[1]
                        || w == p.corners[2]) {
                        continue;
                    }
                    ring_set.insert(w);
                }
            }
        }
        for (Eigen::Index w : ring_set) {
            if (is_bdry[static_cast<std::size_t>(w)]) {
                any_boundary = true;
                break;
            }
        }
        p.ring.assign(ring_set.begin(), ring_set.end());
        std::sort(p.ring.begin(), p.ring.end());
        p.has_boundary = any_boundary;

        patches.push_back(std::move(p));
    }
    return patches;
}

// ---------------------------------------------------------------------------
// Regular box-spline basis: values, gradient, Hessian.
//
// The 12 basis functions of Cirak-Ortiz Eq. (75) / Stam Appendix A are
// quartic polynomials in the barycentric coordinates (u, v, w) with
// u + v + w = 1. To avoid duplicating the coefficient transcription
// across value, gradient, and Hessian routines we factor everything
// through a single 12 x 15 matrix C such that
//
//     N(u, v, w) = (1/12) * C * m4(u, v, w)
//
// where m4 is the vector of 15 degree-4 monomials in (u, v, w). The
// gradient and Hessian then follow from fixed differentiation matrices
// applied to lower-degree monomial vectors m3 and m2:
//
//     d/dv-total = d/dv-explicit - d/du   (since u = 1 - v - w)
//     d/dw-total = d/dw-explicit - d/du
//
// The chain-rule combinations are precomputed as 12 x 10 (gradient)
// and 12 x 6 (Hessian) matrices at first use, so the runtime cost is
// one monomial evaluation plus a few matrix-vector products.
// ---------------------------------------------------------------------------

namespace {

// 15 degree-4 monomials in (u, v, w).
//
// Index layout (lex on (a, b, c) with a + b + c = 4 and a decreasing):
//   0: u^4              5: u^2 w^2          10: v^4
//   1: u^3 v            6: u v^3            11: v^3 w
//   2: u^3 w            7: u v^2 w          12: v^2 w^2
//   3: u^2 v^2          8: u v w^2          13: v w^3
//   4: u^2 v w          9: u w^3            14: w^4
Eigen::Matrix<double, 15, 1> monomials4(double u, double v, double w)
{
    const double u2 = u * u, u3 = u2 * u, u4 = u3 * u;
    const double v2 = v * v, v3 = v2 * v, v4 = v3 * v;
    const double w2 = w * w, w3 = w2 * w, w4 = w3 * w;
    Eigen::Matrix<double, 15, 1> m;
    m(0)  = u4;
    m(1)  = u3 * v;
    m(2)  = u3 * w;
    m(3)  = u2 * v2;
    m(4)  = u2 * v * w;
    m(5)  = u2 * w2;
    m(6)  = u * v3;
    m(7)  = u * v2 * w;
    m(8)  = u * v * w2;
    m(9)  = u * w3;
    m(10) = v4;
    m(11) = v3 * w;
    m(12) = v2 * w2;
    m(13) = v * w3;
    m(14) = w4;
    return m;
}

// 10 degree-3 monomials.
Eigen::Matrix<double, 10, 1> monomials3(double u, double v, double w)
{
    Eigen::Matrix<double, 10, 1> m;
    m(0) = u * u * u;
    m(1) = u * u * v;
    m(2) = u * u * w;
    m(3) = u * v * v;
    m(4) = u * v * w;
    m(5) = u * w * w;
    m(6) = v * v * v;
    m(7) = v * v * w;
    m(8) = v * w * w;
    m(9) = w * w * w;
    return m;
}

// 6 degree-2 monomials.
Eigen::Matrix<double, 6, 1> monomials2(double u, double v, double w)
{
    Eigen::Matrix<double, 6, 1> m;
    m(0) = u * u;
    m(1) = u * v;
    m(2) = u * w;
    m(3) = v * v;
    m(4) = v * w;
    m(5) = w * w;
    return m;
}

// 12 x 15 matrix C such that 12 * N = C * m4. Each row is a basis
// function expressed in the monomials4 basis. Coefficients copied from
// Cirak-Ortiz Eq. (75), cross-checked against the published paper text.
Eigen::Matrix<double, 12, 15> build_basis_coeffs()
{
    Eigen::Matrix<double, 12, 15> C;
    C.setZero();

    // N_1 = u^4 + 2 u^3 v
    C(0, 0) = 1; C(0, 1) = 2;

    // N_2 = u^4 + 2 u^3 w
    C(1, 0) = 1; C(1, 2) = 2;

    // N_3 = u^4 + 2 u^3 w + 6 u^3 v + 6 u^2 v w + 12 u^2 v^2
    //     + 6 u v^2 w + 6 u v^3 + 2 v^3 w + v^4
    C(2, 0) = 1;  C(2, 2) = 2;  C(2, 1) = 6;  C(2, 4) = 6;
    C(2, 3) = 12; C(2, 7) = 6;  C(2, 6) = 6;  C(2, 11) = 2;
    C(2, 10) = 1;

    // N_4 = 6 u^4 + 24 u^3 w + 24 u^2 w^2 + 8 u w^3 + w^4
    //     + 24 u^3 v + 60 u^2 v w + 36 u v w^2 + 6 v w^3
    //     + 24 u^2 v^2 + 36 u v^2 w + 12 v^2 w^2
    //     + 8 u v^3 + 6 v^3 w + v^4
    C(3, 0) = 6;  C(3, 2) = 24; C(3, 5) = 24; C(3, 9) = 8;
    C(3, 14) = 1; C(3, 1) = 24; C(3, 4) = 60; C(3, 8) = 36;
    C(3, 13) = 6; C(3, 3) = 24; C(3, 7) = 36; C(3, 12) = 12;
    C(3, 6) = 8;  C(3, 11) = 6; C(3, 10) = 1;

    // N_5 = u^4 + 6 u^3 w + 12 u^2 w^2 + 6 u w^3 + w^4
    //     + 2 u^3 v + 6 u^2 v w + 6 u v w^2 + 2 v w^3
    C(4, 0) = 1;  C(4, 2) = 6;  C(4, 5) = 12; C(4, 9) = 6;
    C(4, 14) = 1; C(4, 1) = 2;  C(4, 4) = 6;  C(4, 8) = 6;
    C(4, 13) = 2;

    // N_6 = 2 u v^3 + v^4
    C(5, 6) = 2; C(5, 10) = 1;

    // N_7 = u^4 + 6 u^3 w + 12 u^2 w^2 + 6 u w^3 + w^4
    //     + 8 u^3 v + 36 u^2 v w + 36 u v w^2 + 8 v w^3
    //     + 24 u^2 v^2 + 60 u v^2 w + 24 v^2 w^2
    //     + 24 u v^3 + 24 v^3 w + 6 v^4
    C(6, 0) = 1;  C(6, 2) = 6;  C(6, 5) = 12; C(6, 9) = 6;
    C(6, 14) = 1; C(6, 1) = 8;  C(6, 4) = 36; C(6, 8) = 36;
    C(6, 13) = 8; C(6, 3) = 24; C(6, 7) = 60; C(6, 12) = 24;
    C(6, 6) = 24; C(6, 11) = 24; C(6, 10) = 6;

    // N_8 = u^4 + 8 u^3 w + 24 u^2 w^2 + 24 u w^3 + 6 w^4
    //     + 6 u^3 v + 36 u^2 v w + 60 u v w^2 + 24 v w^3
    //     + 12 u^2 v^2 + 36 u v^2 w + 24 v^2 w^2
    //     + 6 u v^3 + 8 v^3 w + v^4
    C(7, 0) = 1;  C(7, 2) = 8;  C(7, 5) = 24; C(7, 9) = 24;
    C(7, 14) = 6; C(7, 1) = 6;  C(7, 4) = 36; C(7, 8) = 60;
    C(7, 13) = 24; C(7, 3) = 12; C(7, 7) = 36; C(7, 12) = 24;
    C(7, 6) = 6;  C(7, 11) = 8; C(7, 10) = 1;

    // N_9 = 2 u w^3 + w^4
    C(8, 9) = 2; C(8, 14) = 1;

    // N_10 = 2 v^3 w + v^4
    C(9, 11) = 2; C(9, 10) = 1;

    // N_11 = 2 u w^3 + w^4 + 6 u v w^2 + 6 v w^3 + 6 u v^2 w
    //      + 12 v^2 w^2 + 2 u v^3 + 6 v^3 w + v^4
    C(10, 9) = 2;  C(10, 14) = 1; C(10, 8) = 6;  C(10, 13) = 6;
    C(10, 7) = 6;  C(10, 12) = 12; C(10, 6) = 2; C(10, 11) = 6;
    C(10, 10) = 1;

    // N_12 = w^4 + 2 v w^3
    C(11, 14) = 1; C(11, 13) = 2;

    return C;
}

// Differentiation matrices D_x4 of size 15 x 10: row k expresses
// d(m4[k])/d(x) as a linear combination of the 10 degree-3 monomials.
// All derivatives are computed treating (u, v, w) as independent.
Eigen::Matrix<double, 15, 10> build_Du4()
{
    Eigen::Matrix<double, 15, 10> D;
    D.setZero();
    D(0, 0) = 4; D(1, 1) = 3; D(2, 2) = 3;
    D(3, 3) = 2; D(4, 4) = 2; D(5, 5) = 2;
    D(6, 6) = 1; D(7, 7) = 1; D(8, 8) = 1; D(9, 9) = 1;
    return D;
}
Eigen::Matrix<double, 15, 10> build_Dv4()
{
    Eigen::Matrix<double, 15, 10> D;
    D.setZero();
    D(1, 0) = 1;
    D(3, 1) = 2; D(4, 2) = 1;
    D(6, 3) = 3; D(7, 4) = 2; D(8, 5) = 1;
    D(10, 6) = 4; D(11, 7) = 3; D(12, 8) = 2; D(13, 9) = 1;
    return D;
}
Eigen::Matrix<double, 15, 10> build_Dw4()
{
    Eigen::Matrix<double, 15, 10> D;
    D.setZero();
    D(2, 0) = 1;
    D(4, 1) = 1; D(5, 2) = 2;
    D(7, 3) = 1; D(8, 4) = 2; D(9, 5) = 3;
    D(11, 6) = 1; D(12, 7) = 2; D(13, 8) = 3; D(14, 9) = 4;
    return D;
}

// Second-derivative matrices of size 15 x 6: row k expresses
// d^2(m4[k])/d(x)d(y) as a linear combination of the 6 degree-2
// monomials, again treating (u, v, w) as independent.
Eigen::Matrix<double, 15, 6> build_Duu4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(0, 0) = 12;
    D(1, 1) = 6; D(2, 2) = 6;
    D(3, 3) = 2; D(4, 4) = 2; D(5, 5) = 2;
    return D;
}
Eigen::Matrix<double, 15, 6> build_Duv4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(1, 0) = 3;
    D(3, 1) = 4; D(4, 2) = 2;
    D(6, 3) = 3; D(7, 4) = 2; D(8, 5) = 1;
    return D;
}
Eigen::Matrix<double, 15, 6> build_Duw4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(2, 0) = 3;
    D(4, 1) = 2; D(5, 2) = 4;
    D(7, 3) = 1; D(8, 4) = 2; D(9, 5) = 3;
    return D;
}
Eigen::Matrix<double, 15, 6> build_Dvv4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(3, 0) = 2;
    D(6, 1) = 6; D(7, 2) = 2;
    D(10, 3) = 12; D(11, 4) = 6; D(12, 5) = 2;
    return D;
}
Eigen::Matrix<double, 15, 6> build_Dvw4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(4, 0) = 1;
    D(7, 1) = 2; D(8, 2) = 2;
    D(11, 3) = 3; D(12, 4) = 4; D(13, 5) = 3;
    return D;
}
Eigen::Matrix<double, 15, 6> build_Dww4()
{
    Eigen::Matrix<double, 15, 6> D;
    D.setZero();
    D(5, 0) = 2;
    D(8, 1) = 2; D(9, 2) = 6;
    D(12, 3) = 2; D(13, 4) = 6; D(14, 5) = 12;
    return D;
}

// Singletons: the basis-coefficient matrix and the precomputed
// 12 x 10 / 12 x 6 chain-rule transforms used at runtime. Each is
// initialised on first call and then reused.
const Eigen::Matrix<double, 12, 15>& basis_coeffs()
{
    static const auto C = build_basis_coeffs();
    return C;
}

// Total ∂N/∂v = (1/12) * C * (Dv4 - Du4) * m3
const Eigen::Matrix<double, 12, 10>& grad_v_op()
{
    static const auto M =
        (basis_coeffs() * (build_Dv4() - build_Du4()) / 12.0).eval();
    return M;
}
// Total ∂N/∂w = (1/12) * C * (Dw4 - Du4) * m3
const Eigen::Matrix<double, 12, 10>& grad_w_op()
{
    static const auto M =
        (basis_coeffs() * (build_Dw4() - build_Du4()) / 12.0).eval();
    return M;
}

// Total Hessian via chain rule with u = 1 - v - w (so d^2 u / d v^2
// etc. all vanish). The combinations follow from
//   d/dv = ∂/∂v - ∂/∂u
// applied twice:
//   d^2/dv^2     = ∂^2/∂u^2 - 2 ∂^2/∂u∂v + ∂^2/∂v^2
//   d^2/dv dw    = ∂^2/∂u^2 - ∂^2/∂u∂v - ∂^2/∂u∂w + ∂^2/∂v∂w
//   d^2/dw^2     = ∂^2/∂u^2 - 2 ∂^2/∂u∂w + ∂^2/∂w^2
const Eigen::Matrix<double, 12, 6>& hess_vv_op()
{
    static const auto M = (basis_coeffs()
        * (build_Duu4() + build_Dvv4() - 2.0 * build_Duv4()) / 12.0).eval();
    return M;
}
const Eigen::Matrix<double, 12, 6>& hess_vw_op()
{
    static const auto M = (basis_coeffs()
        * (build_Duu4() + build_Dvw4() - build_Duv4() - build_Duw4())
        / 12.0).eval();
    return M;
}
const Eigen::Matrix<double, 12, 6>& hess_ww_op()
{
    static const auto M = (basis_coeffs()
        * (build_Duu4() + build_Dww4() - 2.0 * build_Duw4()) / 12.0).eval();
    return M;
}

}  // namespace

Eigen::Matrix<double, 12, 1> regular_basis(double v, double w)
{
    const double u = 1.0 - v - w;
    return basis_coeffs() * monomials4(u, v, w) / 12.0;
}

Eigen::Matrix<double, 12, 2> regular_basis_grad(double v, double w)
{
    const double u = 1.0 - v - w;
    const auto m3 = monomials3(u, v, w);
    Eigen::Matrix<double, 12, 2> G;
    G.col(0) = grad_v_op() * m3;
    G.col(1) = grad_w_op() * m3;
    return G;
}

Eigen::Matrix<double, 12, 3> regular_basis_hess(double v, double w)
{
    const double u = 1.0 - v - w;
    const auto m2 = monomials2(u, v, w);
    Eigen::Matrix<double, 12, 3> H;
    H.col(0) = hess_vv_op() * m2;
    H.col(1) = hess_vw_op() * m2;
    H.col(2) = hess_ww_op() * m2;
    return H;
}

// ---------------------------------------------------------------------------
// Canonical Fig. 9 ordering of a regular interior patch's 1-ring.
// ---------------------------------------------------------------------------

namespace {

// Map directed edge (a, b) -> face index, where (a, b, c) is the CCW
// listing of the face. Each directed edge appears in exactly one face
// for a manifold mesh.
using DirectedEdgeToFace =
    std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index>;

DirectedEdgeToFace build_directed_edge_to_face(const Eigen::MatrixXi& F)
{
    DirectedEdgeToFace m;
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            const Eigen::Index a = F(f, k);
            const Eigen::Index b = F(f, (k + 1) % 3);
            m[{a, b}] = f;
        }
    }
    return m;
}

// Walk CCW around interior vertex P starting from neighbour `start`,
// returning P's neighbours in CCW order. The face containing the
// directed edge (P, current) gives the next spoke as the third vertex.
// For a closed (interior) fan the walk wraps back to `start`; for an
// open (boundary) fan the next directed-edge lookup fails and we
// throw.
std::vector<Eigen::Index> ccw_neighbor_walk(
    Eigen::Index P,
    Eigen::Index start,
    const Eigen::MatrixXi& F,
    const DirectedEdgeToFace& d2f)
{
    std::vector<Eigen::Index> spokes;
    spokes.reserve(8);
    Eigen::Index current = start;
    do {
        spokes.push_back(current);
        const auto it = d2f.find({P, current});
        if (it == d2f.end()) {
            throw std::runtime_error(
                "ccw_neighbor_walk: vertex "
                + std::to_string(static_cast<long long>(P))
                + " has an open fan at neighbour "
                + std::to_string(static_cast<long long>(current))
                + " — boundary patches are not handled by the regular path");
        }
        const Eigen::Index f = it->second;
        Eigen::Index next = -1;
        for (int k = 0; k < 3; ++k) {
            if (F(f, k) == P && F(f, (k + 1) % 3) == current) {
                next = F(f, (k + 2) % 3);
                break;
            }
        }
        // The directed-edge map guarantees we found the right face;
        // F.row(f) must contain (P, current) in CCW order.
        current = next;
    } while (current != start);
    return spokes;
}

// Internal: same semantics as canonical_regular_dofs but with a
// pre-built directed-edge map. Used by assemble_stiffness_augmented to
// hoist d2f construction out of the per-face hot loop (the public
// canonical_regular_dofs would otherwise rebuild d2f for every face,
// turning assembly into O(F^2 log F) — catastrophic on subdivided
// closed meshes with tens of thousands of faces).
std::array<Eigen::Index, 12> canonical_regular_dofs_with_d2f(
    const PatchStencil&         stencil,
    const Eigen::MatrixXi&      F,
    const DirectedEdgeToFace&   d2f);

// Internal: same semantics as gather_stam_patch_dofs but with a
// pre-built directed-edge map. Same rationale as above.
std::vector<Eigen::Index> gather_stam_patch_dofs_with_d2f(
    const PatchStencil&         stencil,
    const Eigen::MatrixXi&      F,
    const DirectedEdgeToFace&   d2f);

}  // namespace

std::vector<Eigen::Index> gather_stam_patch_dofs(
    const PatchStencil&    stencil,
    const Eigen::MatrixXi& F)
{
    return gather_stam_patch_dofs_with_d2f(
        stencil, F, build_directed_edge_to_face(F));
}

namespace {

std::vector<Eigen::Index> gather_stam_patch_dofs_with_d2f(
    const PatchStencil&         stencil,
    const Eigen::MatrixXi&      F,
    const DirectedEdgeToFace&   d2f)
{
    // Identify the extraordinary corner. Exactly one corner must have
    // valence != 6; the other two must be valence-6 (closed fan). This
    // matches the Cirak-Ortiz / Stam configuration after one Loop step.
    int k_ev = -1;
    for (int k = 0; k < 3; ++k) {
        if (stencil.corner_valences[static_cast<std::size_t>(k)] != 6) {
            if (k_ev != -1) {
                throw std::invalid_argument(
                    "gather_stam_patch_dofs: triangle "
                    + std::to_string(static_cast<long long>(
                        stencil.tri_index))
                    + " has more than one corner with valence != 6"
                      " (corners " + std::to_string(k_ev)
                    + " and " + std::to_string(k) + "; one Loop"
                      " subdivision step isolates extraordinary"
                      " vertices, so this should not occur after"
                      " pre-subdivision)");
            }
            k_ev = k;
        }
    }
    if (k_ev == -1) {
        throw std::invalid_argument(
            "gather_stam_patch_dofs: triangle "
            + std::to_string(static_cast<long long>(stencil.tri_index))
            + " has all three corners at valence 6 (regular patch —"
              " use canonical_regular_dofs instead)");
    }
    const int N = stencil.corner_valences[static_cast<std::size_t>(k_ev)];
    if (N < 3) {
        throw std::invalid_argument(
            "gather_stam_patch_dofs: extraordinary vertex has valence "
            + std::to_string(N)
            + " < 3 (degenerate / non-manifold)");
    }
    const Eigen::Index ev = stencil.corners[
        static_cast<std::size_t>(k_ev)];
    const Eigen::Index v1 = stencil.corners[
        static_cast<std::size_t>((k_ev + 1) % 3)];
    const Eigen::Index vN = stencil.corners[
        static_cast<std::size_t>((k_ev + 2) % 3)];

    // CCW walk around ev, starting from v1. ccw_neighbor_walk pivots
    // via directed edge (ev, current); the central face (ev, v1, vN)
    // CCW carries directed edge (ev, v1), so walk[1] is vN. The
    // canonical Stam labelling in build_irregular_patch lays vertex k
    // at angle -2π(k-1)/N (clockwise around ev), so the CCW walk
    // order is [v_1, v_N, v_{N-1}, ..., v_2] — i.e. the CCW order is
    // the canonical labelling REVERSED past slot 1. We re-roll back
    // to canonical slots below.
    const auto walk = ccw_neighbor_walk(ev, v1, F, d2f);
    if (static_cast<int>(walk.size()) != N) {
        throw std::runtime_error(
            "gather_stam_patch_dofs: CCW walk around extraordinary"
            " vertex " + std::to_string(static_cast<long long>(ev))
            + " yielded " + std::to_string(walk.size())
            + " spokes, expected " + std::to_string(N));
    }
    if (walk[1] != vN) {
        throw std::runtime_error(
            "gather_stam_patch_dofs: CCW walk's second spoke ("
            + std::to_string(static_cast<long long>(walk[1]))
            + ") does not match the face's third corner ("
            + std::to_string(static_cast<long long>(vN)) + ")");
    }

    // Third-vertex lookup: face containing directed edge (a, b),
    // return the third corner (not on the edge). The directed-edge map
    // guarantees we find the face.
    auto third = [&](Eigen::Index a, Eigen::Index b) -> Eigen::Index {
        const auto it = d2f.find({a, b});
        if (it == d2f.end()) {
            throw std::runtime_error(
                "gather_stam_patch_dofs: directed edge ("
                + std::to_string(static_cast<long long>(a)) + ", "
                + std::to_string(static_cast<long long>(b))
                + ") not found (boundary patch or non-manifold)");
        }
        const Eigen::Index f = it->second;
        for (int k = 0; k < 3; ++k) {
            if (F(f, k) == a && F(f, (k + 1) % 3) == b) {
                return F(f, (k + 2) % 3);
            }
        }
        return -1;  // unreachable when d2f is consistent with F
    };

    // 1-ring slot mapping (canonical slot k <- walk index):
    //   slot 1     <- walk[0]      = v_1
    //   slot 2     <- walk[N-1]    = v_2 (last CCW neighbour before wrap)
    //   slot 3     <- walk[N-2]
    //   ...
    //   slot k     <- walk[N-k+1]  for k in [2, N]
    //   slot N     <- walk[1]      = v_N
    // Equivalently: slot 1 = walk[0]; slot j+1 for j in [1, N-1] = walk[N-j].
    auto canonical_slot = [&](int slot) -> Eigen::Index {
        // 1-indexed canonical slot j in [1, N].
        if (slot == 1) return walk[0];
        return walk[static_cast<std::size_t>(N - slot + 1)];
    };
    const Eigen::Index v2   = canonical_slot(2);
    const Eigen::Index vNm1 = canonical_slot(N - 1);

    // Outer-ring vertices, layout matches build_irregular_patch in
    // loop_stam.cpp:
    //   N+1 = third(vN, v1)  : across edge (v1, vN) from ev
    //   N+3 = third(v1, v2)  : across edge (v1, v2) from ev
    //   N+5 = third(vNm1, vN): across edge (v_{N-1}, vN) from ev
    //   N+2 = third(N+1, v1) : next CCW outer neighbour of v1 after N+1
    //   N+4 = third(vN, N+1) : next CCW outer neighbour of vN after N+1
    const Eigen::Index Np1 = third(vN, v1);
    const Eigen::Index Np3 = third(v1, v2);
    const Eigen::Index Np5 = third(vNm1, vN);
    const Eigen::Index Np2 = third(Np1, v1);
    const Eigen::Index Np4 = third(vN, Np1);

    const int K = N + 6;
    std::vector<Eigen::Index> dofs(static_cast<std::size_t>(K));
    dofs[0] = ev;
    for (int slot = 1; slot <= N; ++slot) {
        dofs[static_cast<std::size_t>(slot)] = canonical_slot(slot);
    }
    dofs[static_cast<std::size_t>(N + 1)] = Np1;
    dofs[static_cast<std::size_t>(N + 2)] = Np2;
    dofs[static_cast<std::size_t>(N + 3)] = Np3;
    dofs[static_cast<std::size_t>(N + 4)] = Np4;
    dofs[static_cast<std::size_t>(N + 5)] = Np5;
    return dofs;
}

}  // namespace

std::array<Eigen::Index, 12> canonical_regular_dofs(
    const PatchStencil& stencil,
    const Eigen::MatrixXi& F)
{
    return canonical_regular_dofs_with_d2f(
        stencil, F, build_directed_edge_to_face(F));
}

namespace {

std::array<Eigen::Index, 12> canonical_regular_dofs_with_d2f(
    const PatchStencil&         stencil,
    const Eigen::MatrixXi&      F,
    const DirectedEdgeToFace&   d2f)
{
    for (int k = 0; k < 3; ++k) {
        if (stencil.corner_valences[static_cast<std::size_t>(k)] != 6) {
            throw std::invalid_argument(
                "canonical_regular_dofs: corner "
                + std::to_string(k) + " has valence "
                + std::to_string(stencil.corner_valences[
                    static_cast<std::size_t>(k)])
                + " (expected 6)");
        }
    }

    // Walk CCW around each corner, starting from the "next" corner in
    // the F.row(tri) winding. Each walk yields 6 spokes:
    //   walk[0] = next corner
    //   walk[1] = next-next (= prev) corner
    //   walk[2] = edge-opposite of edge to prev corner
    //   walk[3], walk[4] = the 2 outer-ring vertices
    //   walk[5] = edge-opposite of edge to next corner
    const auto walk0 = ccw_neighbor_walk(
        stencil.corners[0], stencil.corners[1], F, d2f);
    const auto walk1 = ccw_neighbor_walk(
        stencil.corners[1], stencil.corners[2], F, d2f);
    const auto walk2 = ccw_neighbor_walk(
        stencil.corners[2], stencil.corners[0], F, d2f);

    if (walk0.size() != 6 || walk1.size() != 6 || walk2.size() != 6) {
        throw std::runtime_error(
            "canonical_regular_dofs: expected 6 spokes per corner walk");
    }

    // Slot table (0-indexed; corresponding to 1-indexed Cirak-Ortiz Fig. 9):
    //   slot 0  = vertex 1   (outer ring around corner 0)
    //   slot 1  = vertex 2   (outer ring around corner 0)
    //   slot 2  = vertex 3   (edge-opposite of edge (corner 0, corner 1))
    //   slot 3  = vertex 4   = corner 0
    //   slot 4  = vertex 5   (edge-opposite of edge (corner 0, corner 2))
    //   slot 5  = vertex 6   (outer ring around corner 1)
    //   slot 6  = vertex 7   = corner 1
    //   slot 7  = vertex 8   = corner 2
    //   slot 8  = vertex 9   (outer ring around corner 2)
    //   slot 9  = vertex 10  (outer ring around corner 1)
    //   slot 10 = vertex 11  (edge-opposite of edge (corner 1, corner 2))
    //   slot 11 = vertex 12  (outer ring around corner 2)
    std::array<Eigen::Index, 12> dofs{};
    dofs[3] = stencil.corners[0];
    dofs[6] = stencil.corners[1];
    dofs[7] = stencil.corners[2];

    // Walk around corner 0, starting from corner 1:
    //   walk0[0] = corners[1]   (slot 6)
    //   walk0[1] = corners[2]   (slot 7)
    //   walk0[2] = edge-opp of (corner 0, corner 2)  -> slot 4
    //   walk0[3] = outer ring (closer to corner 2 side)  -> slot 1
    //   walk0[4] = outer ring (closer to corner 1 side)  -> slot 0
    //   walk0[5] = edge-opp of (corner 0, corner 1)  -> slot 2
    dofs[4] = walk0[2];
    dofs[1] = walk0[3];
    dofs[0] = walk0[4];
    dofs[2] = walk0[5];

    // Walk around corner 1, starting from corner 2:
    //   walk1[0] = corners[2]   (slot 7)
    //   walk1[1] = corners[0]   (slot 3)
    //   walk1[2] = edge-opp of (corner 1, corner 0)  -> slot 2 (already set)
    //   walk1[3] = outer ring around corner 1 (corner-0 side)  -> slot 5
    //   walk1[4] = outer ring around corner 1 (corner-2 side)  -> slot 9
    //   walk1[5] = edge-opp of (corner 1, corner 2)  -> slot 10
    dofs[5]  = walk1[3];
    dofs[9]  = walk1[4];
    dofs[10] = walk1[5];

    // Walk around corner 2, starting from corner 0:
    //   walk2[0] = corners[0]   (slot 3)
    //   walk2[1] = corners[1]   (slot 6)
    //   walk2[2] = edge-opp of (corner 2, corner 1)  -> slot 10 (already set)
    //   walk2[3] = outer ring around corner 2 (corner-1 side)  -> slot 11
    //   walk2[4] = outer ring around corner 2 (corner-0 side)  -> slot 8
    //   walk2[5] = edge-opp of (corner 2, corner 0)  -> slot 4 (already set)
    dofs[11] = walk2[3];
    dofs[8]  = walk2[4];

    return dofs;
}

}  // namespace

// ---------------------------------------------------------------------------
// Limit-surface evaluation at one parametric point.
// ---------------------------------------------------------------------------

PatchEvaluation evaluate_patch_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&               V,
    double                               v,
    double                               w)
{
    if (V.cols() < 3) {
        throw std::invalid_argument(
            "evaluate_patch_regular: V must have at least 3 columns");
    }

    PatchEvaluation pe;
    pe.N      = regular_basis(v, w);
    pe.N_grad = regular_basis_grad(v, w);
    pe.N_hess = regular_basis_hess(v, w);

    // Gather the 12 control points into a fixed 12 x 3 matrix.
    Eigen::Matrix<double, 12, 3> P;
    for (int i = 0; i < 12; ++i) {
        P.row(i) = V.row(canonical_dofs[static_cast<std::size_t>(i)])
                       .head<3>();
    }

    // Position and parametric tangents / second derivatives.
    pe.position      = (P.transpose() * pe.N);
    pe.cov_basis     = P.transpose() * pe.N_grad;
    pe.second_derivs = P.transpose() * pe.N_hess;

    // Surface normal a_3 from the parametric tangents (Cirak-Ortiz
    // Eq. 9). The cross product is non-zero whenever the parametric
    // jacobian is non-singular; we return a unit vector here without
    // checking, leaving the degenerate case for callers.
    pe.normal = pe.cov_basis.col(0).cross(pe.cov_basis.col(1)).normalized();

    return pe;
}

// ---------------------------------------------------------------------------
// Per-element K from one-point centroid quadrature.
// ---------------------------------------------------------------------------

namespace {

// Build the per-quadrature-point K integrand: w * sqrt(a) * (k_L M^T H M + k_B B^T H B).
// Shared between the regular (12-DOF) and irregular (K-DOF) element kernels.
// Computes the local M and B strain matrices, the contravariant elasticity H,
// and accumulates into K_e.
template <int N_DOF, typename MatLike>
void accumulate_stiffness_at_point(
    const Eigen::Vector3d&        a1,
    const Eigen::Vector3d&        a2,
    const Eigen::Vector3d&        a3,
    const Eigen::Vector3d&        a11,
    const Eigen::Vector3d&        a12,
    const Eigen::Vector3d&        a22,
    const Eigen::Matrix<double, N_DOF, 2>& N_grad,
    const Eigen::Matrix<double, N_DOF, 3>& N_hess,
    double                        w_q,
    const ShellMaterial&          material,
    const char*                   caller_name,
    MatLike&                      K_e)
{
    const double cov11 = a1.dot(a1);
    const double cov12 = a1.dot(a2);
    const double cov22 = a2.dot(a2);
    const double a_det = cov11 * cov22 - cov12 * cov12;
    // A non-finite a_det (from a non-finite vertex position or basis
    // value) fails `a_det <= 0.0` (NaN compares false), so guard
    // finiteness explicitly — otherwise sqrt(NaN) propagates into K_e.
    // Mirrors the mass kernel's `!isfinite || <= 0` guard.
    if (!std::isfinite(a_det) || a_det <= 0.0) {
        throw std::runtime_error(
            std::string(caller_name)
            + ": degenerate parametrisation at a quadrature point "
              "(|a_1 x a_2|^2 = "
            + std::to_string(a_det) + ")");
    }
    const double sqrt_a    = std::sqrt(a_det);
    const double inv_det   = 1.0 / a_det;
    const double inv_sqrt_a = 1.0 / sqrt_a;

    // Contravariant metric a^{αβ}.
    const double con11 =  cov22 * inv_det;
    const double con22 =  cov11 * inv_det;
    const double con12 = -cov12 * inv_det;

    // Voigt elasticity H (Cirak-Ortiz Eq. 37).
    const double nu = material.poisson_ratio;
    Eigen::Matrix3d H;
    H(0, 0) = con11 * con11;
    H(1, 1) = con22 * con22;
    H(2, 2) = 0.5 * ((1.0 - nu) * con11 * con22
                     + (1.0 + nu) * con12 * con12);
    H(0, 1) = nu * con11 * con22 + (1.0 - nu) * con12 * con12;
    H(0, 2) = con11 * con12;
    H(1, 2) = con22 * con12;
    H(1, 0) = H(0, 1);
    H(2, 0) = H(0, 2);
    H(2, 1) = H(1, 2);

    const Eigen::Vector3d a2xa3 = a2.cross(a3);
    const Eigen::Vector3d a3xa1 = a3.cross(a1);

    Eigen::Matrix<double, 3, 3 * N_DOF> M = Eigen::Matrix<double, 3, 3 * N_DOF>::Zero();
    Eigen::Matrix<double, 3, 3 * N_DOF> B = Eigen::Matrix<double, 3, 3 * N_DOF>::Zero();

    for (int I = 0; I < N_DOF; ++I) {
        const double dN_dv    = N_grad(I, 0);
        const double dN_dw    = N_grad(I, 1);
        const double d2N_dv2  = N_hess(I, 0);
        const double d2N_dvdw = N_hess(I, 1);
        const double d2N_dw2  = N_hess(I, 2);

        // M^I rows (Eq. 79).
        M.template block<3, 3>(0, 3 * I).row(0) = dN_dv * a1.transpose();
        M.template block<3, 3>(0, 3 * I).row(1) = dN_dw * a2.transpose();
        M.template block<3, 3>(0, 3 * I).row(2) =
            (dN_dw * a1 + dN_dv * a2).transpose();

        // B^I rows (Eq. 80).
        const Eigen::Vector3d N_lap_term = dN_dv * a2xa3 + dN_dw * a3xa1;

        const Eigen::Vector3d B_I_1 =
            -d2N_dv2 * a3
            + inv_sqrt_a * (dN_dv * a11.cross(a2)
                            + dN_dw * a1.cross(a11)
                            + a3.dot(a11) * N_lap_term);
        const Eigen::Vector3d B_I_2 =
            -d2N_dw2 * a3
            + inv_sqrt_a * (dN_dv * a22.cross(a2)
                            + dN_dw * a1.cross(a22)
                            + a3.dot(a22) * N_lap_term);
        const Eigen::Vector3d B_I_3 =
            -d2N_dvdw * a3
            + inv_sqrt_a * (dN_dv * a12.cross(a2)
                            + dN_dw * a1.cross(a12)
                            + a3.dot(a12) * N_lap_term);

        B.template block<3, 3>(0, 3 * I).row(0) = B_I_1.transpose();
        B.template block<3, 3>(0, 3 * I).row(1) = B_I_2.transpose();
        // Voigt convention: 2*beta_12 in the third slot.
        B.template block<3, 3>(0, 3 * I).row(2) = 2.0 * B_I_3.transpose();
    }

    const double scale = w_q * sqrt_a;
    K_e.noalias() += scale * material.k_L * (M.transpose() * H * M);
    K_e.noalias() += scale * material.k_B * (B.transpose() * H * B);
}

}  // namespace

Eigen::Matrix<double, 36, 36> element_stiffness_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&              V_aug,
    const ShellMaterial&                material,
    QuadratureRule                      rule)
{
    // Quadrature on the standard unit triangle (area 1/2). The default
    // SevenPointDunavant rule matches the M assembly's default exactly
    // so K and M are integrated on the same rule. The original
    // Cirak-Ortiz Sec 4.6 recommendation of 1-point centroid preserves
    // convergence ORDER (the asymptotic h -> 0 rate) but leaves
    // per-element O(h^3) errors for the degree-4 integrand B^T H B; at
    // finite h this perturbs eigenvectors of high-m modes visibly (small
    // "bumps" along nodal diameters). 7-point integrates degree-5
    // exactly so the residual drops to O(h^6), well below the FEM
    // truncation floor.
    const auto qp = quadrature_points(rule);

    Eigen::Matrix<double, 36, 36> K_e
        = Eigen::Matrix<double, 36, 36>::Zero();

    for (const auto& q : qp) {
        const auto pe = evaluate_patch_regular(canonical_dofs, V_aug, q.v, q.w);
        accumulate_stiffness_at_point<12>(
            pe.cov_basis.col(0), pe.cov_basis.col(1), pe.normal,
            pe.second_derivs.col(0), pe.second_derivs.col(1), pe.second_derivs.col(2),
            pe.N_grad, pe.N_hess,
            q.weight, material, "element_stiffness_regular", K_e);
    }
    return K_e;
}

// ---------------------------------------------------------------------------
// Per-element M from 7-point Dunavant degree-5 quadrature.
// ---------------------------------------------------------------------------

Eigen::Matrix<double, 36, 36> element_mass_regular(
    const std::array<Eigen::Index, 12>& canonical_dofs,
    const Eigen::MatrixXd&              V_aug,
    double                              surface_density,
    QuadratureRule                      rule)
{
    // Quadrature on the standard unit triangle (area 1/2). Default
    // SevenPointDunavant integrates polynomials of total degree <= 5
    // exactly. The Loop basis is quartic so the product N_I N_J is
    // degree 8 — under SevenPointDunavant the residual under-
    // integration is O(h^6) per element, well below the FEM truncation
    // error. The lower-order rules (1-pt centroid, 3-pt edge-mid) are
    // available for A/B against the original Cirak-Ortiz Sec 4.6
    // baseline and for sensitivity studies; both produce smaller
    // diagonal mass entries than the consistent 7-pt rule because they
    // sample fewer interior points.
    const auto qp = quadrature_points(rule);

    Eigen::Matrix<double, 12, 12> Me_scalar
        = Eigen::Matrix<double, 12, 12>::Zero();

    for (const auto& q : qp) {
        const double v   = q.v;
        const double w   = q.w;
        const double w_q = q.weight;

        const auto pe = evaluate_patch_regular(canonical_dofs, V_aug, v, w);

        // Area element on the limit surface: dA = |a_1 × a_2| dv dw.
        const Eigen::Vector3d a1 = pe.cov_basis.col(0);
        const Eigen::Vector3d a2 = pe.cov_basis.col(1);
        const double sqrt_a = a1.cross(a2).norm();
        if (!std::isfinite(sqrt_a) || sqrt_a <= 0.0) {
            throw std::runtime_error(
                "element_mass_regular: degenerate parametrisation at "
                "(v,w)=(" + std::to_string(v) + "," + std::to_string(w)
                + "), |a1xa2| = " + std::to_string(sqrt_a));
        }

        const double scale = w_q * surface_density * sqrt_a;
        Me_scalar.noalias() += scale * (pe.N * pe.N.transpose());
    }

    // Inflate the 12x12 scalar mass to 36x36 with 3x3 identity blocks
    // per (I, J) pair. The kinetic energy decouples across the three
    // spatial components, so each block M_e(3I:3I+3, 3J:3J+3) is
    // Me_scalar(I, J) * I_3 and off-diagonal spatial entries are zero.
    Eigen::Matrix<double, 36, 36> Me
        = Eigen::Matrix<double, 36, 36>::Zero();
    for (int I = 0; I < 12; ++I) {
        for (int J = 0; J < 12; ++J) {
            const double m_IJ = Me_scalar(I, J);
            for (int d = 0; d < 3; ++d) {
                Me(3 * I + d, 3 * J + d) = m_IJ;
            }
        }
    }
    return Me;
}

// ---------------------------------------------------------------------------
// Schweitzer phantom-vertex augmentation for boundary patches.
// ---------------------------------------------------------------------------

namespace {

// Find the third vertex of a triangle: the one that is not v_a or v_b.
Eigen::Index third_vertex(const Eigen::MatrixXi& F,
                          Eigen::Index           f,
                          Eigen::Index           v_a,
                          Eigen::Index           v_b)
{
    for (int j = 0; j < 3; ++j) {
        const Eigen::Index w = F(f, j);
        if (w != v_a && w != v_b) return w;
    }
    throw std::runtime_error(
        "third_vertex: face does not contain expected edge");
}

}  // namespace

LoopAugmented augment_for_loop_boundary(const Eigen::MatrixXd& V,
                                         const Eigen::MatrixXi& F)
{
    const Eigen::Index n_real       = V.rows();
    const Eigen::Index n_real_faces = F.rows();

    const auto edges = build_edges(F);

    // Index-list of edges that lie on the mesh boundary.
    std::vector<Eigen::Index> bedge_idx;
    bedge_idx.reserve(edges.size() / 4);
    for (Eigen::Index e = 0; e < static_cast<Eigen::Index>(edges.size()); ++e) {
        if (edges[static_cast<std::size_t>(e)].is_boundary()) {
            bedge_idx.push_back(e);
        }
    }
    const Eigen::Index n_phantom =
        static_cast<Eigen::Index>(bedge_idx.size());

    const auto valences = vertex_valences(n_real, edges);
    std::vector<bool> is_bdry(static_cast<std::size_t>(n_real), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }

    // Boundary vertices may have valence 2, 3, or 4. The number of
    // additional "corner phantoms" needed to close the fan to valence 6
    // is (4 - valence): 0 for valence-4 (Schweitzer phantoms close the
    // fan directly), 1 for valence-3 (parallelogram rule), 2 for
    // valence-2 (reflection rule). Valences > 4 (e.g. open-fan corner
    // with > 2 boundary edges incident) and < 2 (degenerate) are
    // rejected.
    Eigen::Index n_phantom_corner = 0;
    for (Eigen::Index v = 0; v < n_real; ++v) {
        if (!is_bdry[static_cast<std::size_t>(v)]) continue;
        const int val = valences[static_cast<std::size_t>(v)];
        if (val == 2 || val == 3 || val == 4) {
            n_phantom_corner += (4 - val);
        } else {
            throw std::invalid_argument(
                "augment_for_loop_boundary: boundary vertex "
                + std::to_string(static_cast<long long>(v))
                + " has valence " + std::to_string(val)
                + " (only valence-2, valence-3, and valence-4 boundary"
                  " vertices are supported)");
        }
    }

    // Phantom positions (Cirak-Ortiz Eq. 54): p = v0 + v1 - v_int,
    // where v_int is the third vertex of the unique boundary face.
    // Also remember the (v0, v1, v_int) triple per phantom for the
    // constraint matrix and for direction classification below.
    struct PhantomMeta {
        Eigen::Index v0;
        Eigen::Index v1;
        Eigen::Index v_int;
        Eigen::Index face_real;     ///< the unique adjacent real face
        bool         left_is_real;  ///< face_left holds the real face
                                    ///<  iff true; otherwise face_right
    };
    // Layout of the augmented vertex array:
    //   rows [0, n_real)                      — original real vertices
    //   rows [n_real, n_real+n_phantom_edge)  — across-edge phantoms
    //   rows [n_real+n_phantom_edge, n_real+n_phantom)
    //                                         — corner phantoms (one
    //                                            per valence-3 corner)
    const Eigen::Index n_phantom_edge = n_phantom;
    const Eigen::Index n_phantom_total = n_phantom_edge + n_phantom_corner;
    std::vector<PhantomMeta> pmeta(static_cast<std::size_t>(n_phantom_edge));

    Eigen::MatrixXd V_aug(n_real + n_phantom_total, V.cols());
    V_aug.topRows(n_real) = V;

    for (Eigen::Index k = 0; k < n_phantom_edge; ++k) {
        const auto& e = edges[static_cast<std::size_t>(
            bedge_idx[static_cast<std::size_t>(k)])];
        const bool left_is_real = (e.face_left  != -1);
        const Eigen::Index f_real = left_is_real ? e.face_left : e.face_right;
        const Eigen::Index v_int = third_vertex(F, f_real, e.v0, e.v1);
        pmeta[static_cast<std::size_t>(k)] =
            {e.v0, e.v1, v_int, f_real, left_is_real};
        V_aug.row(n_real + k) =
            V.row(e.v0) + V.row(e.v1) - V.row(v_int);
    }

    // For each boundary vertex, classify its 2 boundary edges as
    // "out" (real face has v -> other) or "in" (real face has other -> v).
    // The "out" edge appears CCW after the real fan around v; the
    // "in" edge appears at the start of the real fan.
    std::vector<std::vector<std::pair<Eigen::Index, int>>>
        v_to_bedges(static_cast<std::size_t>(n_real));
    for (Eigen::Index k = 0; k < n_phantom_edge; ++k) {
        const auto& m = pmeta[static_cast<std::size_t>(k)];
        const int dir_v0 = m.left_is_real ? +1 : -1;  // +1: v -> other
        const int dir_v1 = m.left_is_real ? -1 : +1;
        v_to_bedges[static_cast<std::size_t>(m.v0)]
            .push_back({k, dir_v0});
        v_to_bedges[static_cast<std::size_t>(m.v1)]
            .push_back({k, dir_v1});
    }

    // Compute per-corner phantom positions, dispatched on valence.
    //
    //  * Valence-4 corners contribute 0 corner phantoms (Schweitzer's
    //    two across-edge phantoms close the fan to valence 6 directly).
    //
    //  * Valence-3 corners (rectangular plate corners after diagonal-
    //    flip): 1 corner phantom via the parallelogram rule
    //        p_corner = p_in + p_out - v = v + b_in + b_out - 2 u,
    //    where u is the unique interior neighbour of the corner. The
    //    constraint coefficients (+1, +1, +1, -2) sum to 1 so the
    //    phantom translates rigidly with the input.
    //
    //  * Valence-2 corners (single-triangle corners — strip endpoints,
    //    raw rectangular plate corners): 2 corner phantoms via the
    //    reflection rule
    //        p_A = 2 v - b_out      (reflects b_out through v)
    //        p_B = 2 v - b_in       (reflects b_in through v)
    //    with constraint coefficients (+2, -1) each, sum 1. Subdivides
    //    the open boundary angle into 3 wedges (between p_in / p_A,
    //    p_A / p_B, p_B / p_out) — three corner-fill triangles instead
    //    of the valence-3 case's two.
    //
    // All corner phantom rows are placed contiguously after the
    // across-edge phantoms in V_aug, in ascending real-vertex order,
    // and within a vertex in (p_A, p_B) order (only p_A used for
    // valence-3).
    struct CornerInfo {
        int           valence;        ///< 2, 3, or 4
        Eigen::Index  k_in;           ///< across-edge phantom index for "in"
        Eigen::Index  k_out;          ///< ditto for "out"
        Eigen::Index  b_in;           ///< boundary neighbour on the "in" side
        Eigen::Index  b_out;          ///< boundary neighbour on the "out" side
        Eigen::Index  u;              ///< interior neighbour (valence-3 only)
        Eigen::Index  pA_row;         ///< first corner phantom row (or -1)
        Eigen::Index  pB_row;         ///< second corner phantom row (valence-2 only, else -1)
    };
    std::vector<CornerInfo> cinfo(static_cast<std::size_t>(n_real));
    for (auto& ci : cinfo) {
        ci = {0, -1, -1, -1, -1, -1, -1, -1};
    }

    auto find_in_out = [&](Eigen::Index v_b)
        -> std::pair<Eigen::Index, Eigen::Index>
    {
        const auto& list = v_to_bedges[static_cast<std::size_t>(v_b)];
        if (list.size() != 2) {
            throw std::runtime_error(
                "augment_for_loop_boundary: vertex "
                + std::to_string(static_cast<long long>(v_b))
                + " has " + std::to_string(list.size())
                + " boundary edges (expected 2 — non-manifold boundary)");
        }
        Eigen::Index k_in = -1, k_out = -1;
        for (const auto& [k, dir] : list) {
            if (dir == +1) k_out = k;
            else           k_in  = k;
        }
        if (k_in == -1 || k_out == -1) {
            throw std::runtime_error(
                "augment_for_loop_boundary: vertex "
                + std::to_string(static_cast<long long>(v_b))
                + " has inconsistent boundary edge directions"
                  " (both \"in\" or both \"out\")");
        }
        return {k_in, k_out};
    };

    auto boundary_other = [](const PhantomMeta& m,
                             Eigen::Index       v_b) -> Eigen::Index
    {
        return (m.v0 == v_b) ? m.v1 : m.v0;
    };

    Eigen::Index next_corner_row = n_real + n_phantom_edge;
    for (Eigen::Index v = 0; v < n_real; ++v) {
        if (!is_bdry[static_cast<std::size_t>(v)]) continue;
        const int val = valences[static_cast<std::size_t>(v)];

        const auto [k_in, k_out] = find_in_out(v);
        const auto& m_in  = pmeta[static_cast<std::size_t>(k_in)];
        const auto& m_out = pmeta[static_cast<std::size_t>(k_out)];
        const Eigen::Index b_in  = boundary_other(m_in,  v);
        const Eigen::Index b_out = boundary_other(m_out, v);

        auto& ci = cinfo[static_cast<std::size_t>(v)];
        ci.valence = val;
        ci.k_in    = k_in;
        ci.k_out   = k_out;
        ci.b_in    = b_in;
        ci.b_out   = b_out;

        if (val == 4) {
            // No corner phantoms; fields populated above are enough
            // for the gap-fill loop below.
            continue;
        }
        if (val == 3) {
            // Valence-3: one corner phantom (parallelogram rule).
            // Both incident triangles share the unique interior u.
            if (m_in.v_int != m_out.v_int) {
                throw std::runtime_error(
                    "augment_for_loop_boundary: valence-3 corner "
                    + std::to_string(static_cast<long long>(v))
                    + " has different third vertices on its two incident"
                      " boundary triangles (mesh non-manifold)");
            }
            ci.u      = m_in.v_int;
            ci.pA_row = next_corner_row++;
            V_aug.row(ci.pA_row) =
                V.row(v) + V.row(b_in) + V.row(b_out) - 2.0 * V.row(ci.u);
        } else /* val == 2 (validated above) */ {
            // Valence-2: two corner phantoms via reflection rule.
            // The single incident triangle has b_in and b_out as the
            // other two corners, so m_in.v_int == b_out and
            // m_out.v_int == b_in (sanity assertion).
            if (m_in.v_int != b_out || m_out.v_int != b_in) {
                throw std::runtime_error(
                    "augment_for_loop_boundary: valence-2 corner "
                    + std::to_string(static_cast<long long>(v))
                    + " incident-triangle topology is inconsistent");
            }
            ci.pA_row = next_corner_row++;
            ci.pB_row = next_corner_row++;
            // p_A reflects b_out through v; p_B reflects b_in.
            V_aug.row(ci.pA_row) = 2.0 * V.row(v) - V.row(b_out);
            V_aug.row(ci.pB_row) = 2.0 * V.row(v) - V.row(b_in);
        }
    }

    // Sparse constraint matrix. Top n_real rows: identity. Next
    // n_phantom_edge rows: across-edge phantom (Cirak-Ortiz Eq. 54),
    // each (+1, +1, -1) on (v0, v1, v_int). Corner phantom rows
    // dispatch on valence:
    //   val 3:  (+1, +1, +1, -2) on (v, b_in, b_out, u)   [parallelogram]
    //   val 2:  p_A: (+2, -1) on (v, b_out)               [reflection]
    //           p_B: (+2, -1) on (v, b_in)
    // Coefficients sum to 1 in every phantom case so the augmented
    // mesh translates rigidly with the input.
    const Eigen::Index dim_aug  = 3 * (n_real + n_phantom_total);
    const Eigen::Index dim_real = 3 * n_real;
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(
        3 * n_real + 9 * n_phantom_edge + 12 * n_phantom_corner));
    for (Eigen::Index v = 0; v < n_real; ++v) {
        for (int d = 0; d < 3; ++d) {
            trips.emplace_back(3 * v + d, 3 * v + d, 1.0);
        }
    }
    for (Eigen::Index k = 0; k < n_phantom_edge; ++k) {
        const auto& m = pmeta[static_cast<std::size_t>(k)];
        for (int d = 0; d < 3; ++d) {
            const Eigen::Index r = 3 * (n_real + k) + d;
            trips.emplace_back(r, 3 * m.v0    + d,  1.0);
            trips.emplace_back(r, 3 * m.v1    + d,  1.0);
            trips.emplace_back(r, 3 * m.v_int + d, -1.0);
        }
    }
    for (Eigen::Index v = 0; v < n_real; ++v) {
        const auto& ci = cinfo[static_cast<std::size_t>(v)];
        if (ci.valence == 3) {
            for (int d = 0; d < 3; ++d) {
                const Eigen::Index r = 3 * ci.pA_row + d;
                trips.emplace_back(r, 3 * v        + d,  1.0);
                trips.emplace_back(r, 3 * ci.b_in  + d,  1.0);
                trips.emplace_back(r, 3 * ci.b_out + d,  1.0);
                trips.emplace_back(r, 3 * ci.u     + d, -2.0);
            }
        } else if (ci.valence == 2) {
            for (int d = 0; d < 3; ++d) {
                const Eigen::Index rA = 3 * ci.pA_row + d;
                trips.emplace_back(rA, 3 * v        + d,  2.0);
                trips.emplace_back(rA, 3 * ci.b_out + d, -1.0);
                const Eigen::Index rB = 3 * ci.pB_row + d;
                trips.emplace_back(rB, 3 * v        + d,  2.0);
                trips.emplace_back(rB, 3 * ci.b_in  + d, -1.0);
            }
        }
    }
    Eigen::SparseMatrix<double> C(dim_aug, dim_real);
    C.setFromTriplets(trips.begin(), trips.end());

    // Phantom triangles. The "across-edge" phantom triangle for
    // boundary edge {v0, v1} carries the directed edge OPPOSITE to the
    // one used by the real face, so the augmented mesh's
    // directed-edge-to-face map remains 1-to-1.
    //   real has v0 -> v1  =>  phantom winding (v1, v0, p)
    //   real has v1 -> v0  =>  phantom winding (v0, v1, p)
    std::vector<std::array<Eigen::Index, 3>> phantom_faces;
    phantom_faces.reserve(static_cast<std::size_t>(
        2 * n_phantom_edge + n_phantom_corner));
    for (Eigen::Index k = 0; k < n_phantom_edge; ++k) {
        const auto& m = pmeta[static_cast<std::size_t>(k)];
        const Eigen::Index p = n_real + k;
        if (m.left_is_real) {
            phantom_faces.push_back({m.v1, m.v0, p});
        } else {
            phantom_faces.push_back({m.v0, m.v1, p});
        }
    }

    // Per-vertex gap-closing triangles around boundary corners. The
    // gap is the open boundary angle between p_in and p_out (the two
    // across-edge phantoms incident to v). The fan around v in CCW
    // order is:
    //   ... real triangles ... -> p_in -> [corner phantoms] -> p_out -> ...
    // and we emit (5 - valence) corner-fill triangles (1, 2, or 3
    // depending on valence).
    for (Eigen::Index v = 0; v < n_real; ++v) {
        if (!is_bdry[static_cast<std::size_t>(v)]) continue;
        const auto& ci = cinfo[static_cast<std::size_t>(v)];
        const Eigen::Index p_in  = n_real + ci.k_in;
        const Eigen::Index p_out = n_real + ci.k_out;
        if (ci.valence == 4) {
            phantom_faces.push_back({v, p_in, p_out});
        } else if (ci.valence == 3) {
            phantom_faces.push_back({v, p_in,      ci.pA_row});
            phantom_faces.push_back({v, ci.pA_row, p_out});
        } else /* ci.valence == 2 */ {
            phantom_faces.push_back({v, p_in,      ci.pA_row});
            phantom_faces.push_back({v, ci.pA_row, ci.pB_row});
            phantom_faces.push_back({v, ci.pB_row, p_out});
        }
    }

    Eigen::MatrixXi F_aug(
        n_real_faces + static_cast<Eigen::Index>(phantom_faces.size()), 3);
    F_aug.topRows(n_real_faces) = F;
    for (Eigen::Index k = 0;
         k < static_cast<Eigen::Index>(phantom_faces.size()); ++k)
    {
        const auto& tri = phantom_faces[static_cast<std::size_t>(k)];
        F_aug(n_real_faces + k, 0) = static_cast<int>(tri[0]);
        F_aug(n_real_faces + k, 1) = static_cast<int>(tri[1]);
        F_aug(n_real_faces + k, 2) = static_cast<int>(tri[2]);
    }

    LoopAugmented out;
    out.V_aug           = std::move(V_aug);
    out.F_aug           = std::move(F_aug);
    out.C               = std::move(C);
    out.n_real          = n_real;
    out.n_real_faces    = n_real_faces;
    out.n_phantom       = n_phantom_total;
    out.n_phantom_faces =
        static_cast<Eigen::Index>(phantom_faces.size());
    return out;
}

// ---------------------------------------------------------------------------
// One global Loop subdivision step.
//
// Implements the four Loop masks. Each new subdivided vertex is a
// sparse linear combination of the input vertices, captured both in
// the materialised V_sub matrix and in the sparse constraint matrix S.
// ---------------------------------------------------------------------------

namespace {

/// Loop's interior smoothing weight β_n (Loop 1987).
double loop_beta(int n)
{
    if (n == 3) return 3.0 / 16.0;
    const double c = 0.375
        + 0.25 * std::cos(2.0 * std::numbers::pi / static_cast<double>(n));
    return (1.0 / static_cast<double>(n)) * (0.625 - c * c);
}

}  // namespace

LoopSubdivision loop_subdivide_one_step(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    const Eigen::Index n_real       = V.rows();
    const Eigen::Index n_real_faces = F.rows();

    const auto edges    = build_edges(F);
    const Eigen::Index n_edges = static_cast<Eigen::Index>(edges.size());
    const Eigen::Index n_sub   = n_real + n_edges;

    const auto valences = vertex_valences(n_real, edges);

    // Boundary classification + per-boundary-vertex list of its (up to
    // two) boundary neighbours. The boundary even rule reads only the
    // boundary neighbours, never the interior ones, so we keep them
    // separate from the all-neighbours adjacency.
    std::vector<bool> is_bdry(static_cast<std::size_t>(n_real), false);
    std::vector<std::vector<Eigen::Index>>
        bdry_neighbors(static_cast<std::size_t>(n_real));
    std::vector<std::vector<Eigen::Index>>
        all_neighbors(static_cast<std::size_t>(n_real));
    for (const auto& e : edges) {
        all_neighbors[static_cast<std::size_t>(e.v0)].push_back(e.v1);
        all_neighbors[static_cast<std::size_t>(e.v1)].push_back(e.v0);
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
            bdry_neighbors[static_cast<std::size_t>(e.v0)].push_back(e.v1);
            bdry_neighbors[static_cast<std::size_t>(e.v1)].push_back(e.v0);
        }
    }

    // Edge index lookup keyed by the canonical (v0, v1) pair with v0 < v1
    // (matching build_edges' storage convention).
    std::map<std::pair<Eigen::Index, Eigen::Index>, Eigen::Index>
        edge_index_map;
    for (Eigen::Index e = 0; e < n_edges; ++e) {
        const auto& edge = edges[static_cast<std::size_t>(e)];
        edge_index_map[{edge.v0, edge.v1}] = e;
    }
    auto edge_idx = [&edge_index_map](Eigen::Index a,
                                      Eigen::Index b) -> Eigen::Index {
        if (a > b) std::swap(a, b);
        const auto it = edge_index_map.find({a, b});
        if (it == edge_index_map.end()) {
            throw std::runtime_error(
                "loop_subdivide_one_step: edge ("
                + std::to_string(static_cast<long long>(a)) + ", "
                + std::to_string(static_cast<long long>(b))
                + ") not in the edge list");
        }
        return it->second;
    };

    // Triplets for the sparse 3*n_sub x 3*n_real constraint matrix S.
    // Per scalar (Loop) weight w from sub-vertex i to real-vertex j we
    // emit 3 triplets — one per spatial component — keeping the three
    // components diagonal in the (sub, real) block.
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(
        3 * (8 * n_real + 4 * n_edges)));
    auto add_weight = [&trips](Eigen::Index sub_v,
                               Eigen::Index real_v,
                               double       w) {
        for (int d = 0; d < 3; ++d) {
            trips.emplace_back(3 * sub_v + d, 3 * real_v + d, w);
        }
    };

    // -- Even rule: top n_real rows. --
    for (Eigen::Index v = 0; v < n_real; ++v) {
        const auto& bn = bdry_neighbors[static_cast<std::size_t>(v)];
        if (is_bdry[static_cast<std::size_t>(v)]) {
            // Boundary vertex must have exactly 2 boundary neighbours
            // (manifold boundary). Strip-style valence-2 / corner-style
            // valence-3 boundary vertices satisfy this constraint;
            // valence-N corners with > 2 boundary edges do not and are
            // rejected here as they are by augment_for_loop_boundary.
            if (bn.size() != 2) {
                throw std::runtime_error(
                    "loop_subdivide_one_step: boundary vertex "
                    + std::to_string(static_cast<long long>(v))
                    + " has " + std::to_string(bn.size())
                    + " boundary edges (expected 2 for a manifold boundary)");
            }
            // Standard cubic B-spline boundary mask (3/4, 1/8, 1/8),
            // applied UNIFORMLY to every boundary vertex — valence-2 strip
            // ends and valence-3 corners included. This smooths the boundary
            // as a cubic curve rather than preserving sharp corners; vanilla
            // Loop has no corner/crease rule, and corner-rounding is the
            // intended limit behaviour here (it is part of why the Loop path
            // is the robust shipping method on disks/plates — the modal gates
            // are tuned to it). A crease rule that pins valence-2/3 corners
            // could be added if sharp-feature preservation is ever needed.
            add_weight(v, v, 0.75);
            add_weight(v, bn[0], 0.125);
            add_weight(v, bn[1], 0.125);
        } else {
            const int n = valences[static_cast<std::size_t>(v)];
            const double beta = loop_beta(n);
            add_weight(v, v, 1.0 - n * beta);
            for (Eigen::Index nb : all_neighbors[static_cast<std::size_t>(v)]) {
                add_weight(v, nb, beta);
            }
        }
    }

    // -- Odd rule: rows n_real..n_sub-1. --
    for (Eigen::Index e = 0; e < n_edges; ++e) {
        const Eigen::Index sub_v = n_real + e;
        const auto& edge = edges[static_cast<std::size_t>(e)];
        if (edge.is_boundary()) {
            add_weight(sub_v, edge.v0, 0.5);
            add_weight(sub_v, edge.v1, 0.5);
        } else {
            const Eigen::Index c_left =
                third_vertex(F, edge.face_left,  edge.v0, edge.v1);
            const Eigen::Index c_right =
                third_vertex(F, edge.face_right, edge.v0, edge.v1);
            add_weight(sub_v, edge.v0,  0.375);
            add_weight(sub_v, edge.v1,  0.375);
            add_weight(sub_v, c_left,   0.125);
            add_weight(sub_v, c_right,  0.125);
        }
    }

    Eigen::SparseMatrix<double> S(3 * n_sub, 3 * n_real);
    S.setFromTriplets(trips.begin(), trips.end());

    // Materialise V_sub by replaying the same scalar weights against
    // the input positions. Using the triplets directly (rather than
    // S * V_flat) avoids a sparse-dense product and the repeated
    // 3-component pattern: each scalar weight w from sub_v to real_v
    // contributes w * V.row(real_v) to V_sub.row(sub_v) once.
    Eigen::MatrixXd V_sub(n_sub, V.cols());
    V_sub.setZero();
    for (const auto& trip : trips) {
        const Eigen::Index sub_row    = trip.row() / 3;
        const Eigen::Index spatial    = trip.row() % 3;
        const Eigen::Index spatial_in = trip.col() % 3;
        if (spatial != spatial_in) continue;
        const Eigen::Index real_row = trip.col() / 3;
        if (spatial == 0) {
            V_sub.row(sub_row) += trip.value() * V.row(real_row);
        }
    }

    // Build F_sub: 4 sub-triangles per original face. Winding follows
    // the parent so the medial-triangle and corner-triangles are CCW
    // when the parent is CCW.
    Eigen::MatrixXi F_sub(4 * n_real_faces, 3);
    for (Eigen::Index f = 0; f < n_real_faces; ++f) {
        const Eigen::Index a = F(f, 0);
        const Eigen::Index b = F(f, 1);
        const Eigen::Index c = F(f, 2);
        const Eigen::Index e_ab = n_real + edge_idx(a, b);
        const Eigen::Index e_bc = n_real + edge_idx(b, c);
        const Eigen::Index e_ca = n_real + edge_idx(c, a);
        F_sub(4 * f + 0, 0) = static_cast<int>(a);
        F_sub(4 * f + 0, 1) = static_cast<int>(e_ab);
        F_sub(4 * f + 0, 2) = static_cast<int>(e_ca);
        F_sub(4 * f + 1, 0) = static_cast<int>(e_ab);
        F_sub(4 * f + 1, 1) = static_cast<int>(b);
        F_sub(4 * f + 1, 2) = static_cast<int>(e_bc);
        F_sub(4 * f + 2, 0) = static_cast<int>(e_ca);
        F_sub(4 * f + 2, 1) = static_cast<int>(e_bc);
        F_sub(4 * f + 2, 2) = static_cast<int>(c);
        F_sub(4 * f + 3, 0) = static_cast<int>(e_ab);
        F_sub(4 * f + 3, 1) = static_cast<int>(e_bc);
        F_sub(4 * f + 3, 2) = static_cast<int>(e_ca);
    }

    LoopSubdivision out;
    out.V_sub            = std::move(V_sub);
    out.F_sub            = std::move(F_sub);
    out.S                = std::move(S);
    out.n_real           = n_real;
    out.n_real_faces     = n_real_faces;
    out.n_edge_midpoints = n_edges;
    return out;
}

LoopSubdivision loop_subdivide_n_times(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int                    n_passes)
{
    if (n_passes < 0) {
        throw std::invalid_argument(
            "loop_subdivide_n_times: n_passes must be >= 0 (got "
            + std::to_string(n_passes) + ")");
    }

    if (n_passes == 0) {
        // Identity case: no subdivision. S is the identity sparse
        // matrix, F_sub == F, V_sub == V. n_edge_midpoints = 0.
        const Eigen::Index n_v = V.rows();
        Eigen::SparseMatrix<double> S(3 * n_v, 3 * n_v);
        S.setIdentity();
        LoopSubdivision out;
        out.V_sub            = V;
        out.F_sub            = F;
        out.S                = std::move(S);
        out.n_real           = n_v;
        out.n_real_faces     = F.rows();
        out.n_edge_midpoints = 0;
        return out;
    }

    LoopSubdivision sub = loop_subdivide_one_step(V, F);
    for (int k = 1; k < n_passes; ++k) {
        LoopSubdivision next = loop_subdivide_one_step(sub.V_sub, sub.F_sub);
        // Compose constraint matrices. Both are 3·(n_intermediate) ×
        // 3·(n_real_at_input_to_pass), and we want to fold each pass
        // back to the *original* DOFs.
        next.S = (next.S * sub.S).eval();
        // Original-mesh totals stay fixed across all passes.
        next.n_real       = sub.n_real;
        next.n_real_faces = sub.n_real_faces;
        sub = std::move(next);
    }
    // The single-pass "rows [n_real, n_real + n_edge_midpoints) are the new
    // edge-midpoint vertices" layout contract only holds for one pass. After
    // several passes V_sub interleaves multiple subdivision generations, so
    // the count from the most recent pass is meaningless as a layout offset —
    // flag it with a -1 sentinel rather than leave a misleading number.
    if (n_passes > 1) {
        sub.n_edge_midpoints = -1;
    }
    return sub;
}

// ---------------------------------------------------------------------------
// Global stiffness assembly on the augmented mesh.
// ---------------------------------------------------------------------------

Eigen::SparseMatrix<double> assemble_stiffness_augmented(
    const LoopAugmented& aug,
    const ShellMaterial& material,
    bool                 skip_irregular_triangles,
    bool                 use_stam_for_irregular,
    QuadratureRule       rule)
{
    const Eigen::Index n_aug   = aug.V_aug.rows();
    const Eigen::Index dim_aug = 3 * n_aug;

    // Build patch stencils on the augmented mesh. For real triangles
    // whose original corners were on the boundary, the corners now
    // have augmented valence 6 (4 real + 2 phantom edges) and the
    // 1-ring is 12 vertices (some real, some phantom). For triangles
    // entirely interior in F, F_aug doesn't change anything and the
    // stencil is the same as in F.
    const auto patches = build_patch_stencils(aug.V_aug, aug.F_aug);

    // Hoist the directed-edge -> face map out of the per-face hot loop.
    // canonical_regular_dofs and gather_stam_patch_dofs both rebuild
    // d2f on every call by default; with O(F) faces and O(F log F)
    // d2f construction the assembly degenerates to O(F^2 log F) —
    // unusable past ~10k subdivided faces (genus-3 surface stalls
    // for minutes). Building d2f once and threading it through the
    // _with_d2f variants is O(F log F) total.
    const auto d2f = build_directed_edge_to_face(aug.F_aug);

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(36 * 36 * aug.n_real_faces));

    // Cache StamEvaluator per valence so the eigenstructure / picking
    // matrices are computed once per N (Stam S.7 plan, "Caching").
    std::map<int, StamEvaluator> stam_cache;

    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        const auto& p = patches[static_cast<std::size_t>(f)];

        // Classify the face: count irregular corners.
        int n_irregular = 0;
        int kev = -1;
        for (int k = 0; k < 3; ++k) {
            if (p.corner_valences[static_cast<std::size_t>(k)] != 6) {
                ++n_irregular;
                kev = k;
            }
        }

        if (n_irregular == 0) {
            // Regular path — canonical Fig. 9 ordering. Phantom DOFs
            // may appear in the returned indices alongside real ones;
            // that's fine because the limit-surface evaluation reads
            // positions from V_aug (which holds both) and the
            // constraint matrix C will reduce phantom DOFs onto real
            // DOFs after assembly.
            const auto dofs = canonical_regular_dofs_with_d2f(
                p, aug.F_aug, d2f);

            // Reference-triangle quadrature per @p rule. Default
            // (7-point Dunavant) matches the M assembly's default,
            // deviating from Cirak-Ortiz Sec 4.6's 1-point centroid —
            // the asymptotic convergence order is unchanged but the
            // per-element residual drops from O(h^3) to O(h^6),
            // removing high-m mode-shape bumps visible at typical
            // resolutions. The 1-pt and 3-pt rules are available for
            // direct A/B against the original Cirak-Ortiz baseline.
            const auto K_e = element_stiffness_regular(
                dofs, aug.V_aug, material, rule);

            // Scatter K_e into K_aug. Slot ordering of K_e is canonical
            // Fig. 9; mapping slot I to the global augmented DOF block
            // is dofs[I] (3 spatial components per slot).
            for (int I = 0; I < 12; ++I) {
                const Eigen::Index gi = dofs[
                    static_cast<std::size_t>(I)];
                for (int J = 0; J < 12; ++J) {
                    const Eigen::Index gj = dofs[
                        static_cast<std::size_t>(J)];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = K_e(3 * I + di, 3 * J + dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3 * gi + di, 3 * gj + dj, v);
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Irregular face — three policies.
        if (use_stam_for_irregular && n_irregular == 1) {
            const int N = p.corner_valences[
                static_cast<std::size_t>(kev)];

            // Gather K = N+6 canonical Stam DOFs.
            const auto dofs = gather_stam_patch_dofs_with_d2f(
                p, aug.F_aug, d2f);

            // Build (or look up) the cached evaluator for this valence.
            auto it = stam_cache.find(N);
            if (it == stam_cache.end()) {
                it = stam_cache.emplace(
                    N, make_stam_evaluator(N)).first;
            }
            const StamEvaluator& ev = it->second;

            // Reference-triangle quadrature per @p rule on the central
            // irregular sub-tile, matching the regular path's K assembly
            // and the M assembly. element_stiffness_stam evaluates the
            // Stam basis at each quadrature point internally; all sample
            // points of every supported rule are safely inside the
            // medial tile and away from the extraordinary-vertex apex.
            const Eigen::MatrixXd K_e = element_stiffness_stam(
                ev, dofs, aug.V_aug, material, rule);

            const int K = N + 6;
            for (int I = 0; I < K; ++I) {
                const Eigen::Index gi = dofs[
                    static_cast<std::size_t>(I)];
                for (int J = 0; J < K; ++J) {
                    const Eigen::Index gj = dofs[
                        static_cast<std::size_t>(J)];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = K_e(3 * I + di, 3 * J + dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3 * gi + di, 3 * gj + dj, v);
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Fall-through: skip or throw.
        if (skip_irregular_triangles) {
            continue;
        }
        throw std::runtime_error(
            "assemble_stiffness_augmented: real triangle "
            + std::to_string(static_cast<long long>(f))
            + " has " + std::to_string(n_irregular)
            + " corner(s) with valence != 6"
              " (pass skip_irregular_triangles=true to drop this"
              " contribution under the Cirak-Ortiz one-pass"
              " subdivision approximation, or use_stam_for_irregular=true"
              " for the S-series Stam exact evaluation — the latter"
              " requires exactly one extraordinary corner per face"
              " after Loop subdivision)");
    }

    Eigen::SparseMatrix<double> K_aug(dim_aug, dim_aug);
    K_aug.setFromTriplets(trips.begin(), trips.end());
    return K_aug;
}

Eigen::SparseMatrix<double> assemble_mass_augmented(
    const LoopAugmented& aug,
    double               surface_density,
    bool                 skip_irregular_triangles,
    bool                 use_stam_for_irregular,
    QuadratureRule       rule)
{
    const Eigen::Index n_aug   = aug.V_aug.rows();
    const Eigen::Index dim_aug = 3 * n_aug;

    const auto patches = build_patch_stencils(aug.V_aug, aug.F_aug);
    const auto d2f     = build_directed_edge_to_face(aug.F_aug);

    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(36 * 36 * aug.n_real_faces));

    std::map<int, StamEvaluator> stam_cache;

    for (Eigen::Index f = 0; f < aug.n_real_faces; ++f) {
        const auto& p = patches[static_cast<std::size_t>(f)];

        int n_irregular = 0;
        int kev = -1;
        for (int k = 0; k < 3; ++k) {
            if (p.corner_valences[static_cast<std::size_t>(k)] != 6) {
                ++n_irregular;
                kev = k;
            }
        }

        if (n_irregular == 0) {
            const auto dofs = canonical_regular_dofs_with_d2f(
                p, aug.F_aug, d2f);
            const auto M_e = element_mass_regular(
                dofs, aug.V_aug, surface_density, rule);

            for (int I = 0; I < 12; ++I) {
                const Eigen::Index gi = dofs[
                    static_cast<std::size_t>(I)];
                for (int J = 0; J < 12; ++J) {
                    const Eigen::Index gj = dofs[
                        static_cast<std::size_t>(J)];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = M_e(3 * I + di, 3 * J + dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3 * gi + di, 3 * gj + dj, v);
                            }
                        }
                    }
                }
            }
            continue;
        }

        if (use_stam_for_irregular && n_irregular == 1) {
            const int N = p.corner_valences[
                static_cast<std::size_t>(kev)];
            const auto dofs = gather_stam_patch_dofs_with_d2f(
                p, aug.F_aug, d2f);

            auto it = stam_cache.find(N);
            if (it == stam_cache.end()) {
                it = stam_cache.emplace(
                    N, make_stam_evaluator(N)).first;
            }
            const StamEvaluator& ev = it->second;

            const Eigen::MatrixXd M_e = element_mass_stam(
                ev, dofs, aug.V_aug, surface_density, rule);

            const int K = N + 6;
            for (int I = 0; I < K; ++I) {
                const Eigen::Index gi = dofs[
                    static_cast<std::size_t>(I)];
                for (int J = 0; J < K; ++J) {
                    const Eigen::Index gj = dofs[
                        static_cast<std::size_t>(J)];
                    for (int di = 0; di < 3; ++di) {
                        for (int dj = 0; dj < 3; ++dj) {
                            const double v = M_e(3 * I + di, 3 * J + dj);
                            if (v != 0.0) {
                                trips.emplace_back(
                                    3 * gi + di, 3 * gj + dj, v);
                            }
                        }
                    }
                }
            }
            continue;
        }

        if (skip_irregular_triangles) {
            continue;
        }
        throw std::runtime_error(
            "assemble_mass_augmented: real triangle "
            + std::to_string(static_cast<long long>(f))
            + " has " + std::to_string(n_irregular)
            + " corner(s) with valence != 6"
              " (pass skip_irregular_triangles=true to drop, or"
              " use_stam_for_irregular=true for Stam exact mass —"
              " mirrors the assemble_stiffness_augmented policy so K and"
              " M drop / Stam-evaluate the same set of sub-triangles)");
    }

    Eigen::SparseMatrix<double> M_aug(dim_aug, dim_aug);
    M_aug.setFromTriplets(trips.begin(), trips.end());
    return M_aug;
}

Eigen::SparseMatrix<double> assemble_mass_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double                 surface_density,
    int                    n_passes,
    bool                   use_stam,
    QuadratureRule         rule)
{
    if (n_passes < 1) {
        throw std::invalid_argument(
            "assemble_mass_loop: n_passes must be >= 1 (got "
            + std::to_string(n_passes) + ")");
    }
    if (surface_density <= 0.0) {
        throw std::invalid_argument(
            "assemble_mass_loop: surface_density must be > 0 (got "
            + std::to_string(surface_density) + ")");
    }

    // Mirror assemble_stiffness_loop: same extraordinary-vertex detection,
    // same subdivision-or-fast-path dispatch, same C / S reductions. K
    // and M *must* share these structural choices — if K uses Stam on a
    // sub-triangle and M drops it (or vice versa), the generalized
    // eigenproblem becomes ill-defined.
    const auto edges    = build_edges(F);
    const auto valences = vertex_valences(V.rows(), edges);
    std::vector<bool> is_bdry(static_cast<std::size_t>(V.rows()), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }
    bool any_extraordinary = false;
    for (Eigen::Index v = 0; v < V.rows(); ++v) {
        if (!is_bdry[static_cast<std::size_t>(v)]
            && valences[static_cast<std::size_t>(v)] != 6)
        {
            any_extraordinary = true;
            break;
        }
    }

    Eigen::SparseMatrix<double> M;
    if (!any_extraordinary) {
        const auto aug   = augment_for_loop_boundary(V, F);
        const auto M_aug = assemble_mass_augmented(
            aug, surface_density,
            /*skip_irregular_triangles=*/false,
            /*use_stam_for_irregular=*/false,
            rule);
        Eigen::SparseMatrix<double> Ct = aug.C.transpose();
        M = Ct * M_aug * aug.C;
    } else {
        const auto sub = loop_subdivide_n_times(V, F, n_passes);
        const auto aug = augment_for_loop_boundary(sub.V_sub, sub.F_sub);
        const auto M_sub_aug = assemble_mass_augmented(
            aug, surface_density,
            /*skip_irregular_triangles=*/!use_stam,
            /*use_stam_for_irregular=*/use_stam,
            rule);
        Eigen::SparseMatrix<double> Ct    = aug.C.transpose();
        Eigen::SparseMatrix<double> M_sub = Ct * M_sub_aug * aug.C;
        Eigen::SparseMatrix<double> St    = sub.S.transpose();
        M = St * M_sub * sub.S;
    }

    // Belt-and-suspenders finiteness sweep on the assembled stream, the
    // mass-side twin of the guard in assemble_stiffness_loop. A non-finite
    // M flows into solve_modal_eigenproblem_with_rigid_filter (factorises
    // K - sigma*M, LLT-factors the rigid-body Gram Vr^T M Vr) as an opaque
    // factorisation failure or silent garbage modes; reject it here.
    M.makeCompressed();
    if (!M.coeffs().allFinite()) {
        throw std::runtime_error(
            "assemble_mass_loop: assembled mass matrix contains a "
            "non-finite entry (NaN/Inf) — likely a non-finite vertex "
            "position or degenerate geometry upstream.");
    }
    return M;
}

Eigen::SparseMatrix<double> assemble_stiffness_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ShellMaterial&   material,
    int                    n_passes,
    bool                   use_stam,
    QuadratureRule         rule)
{
    if (n_passes < 1) {
        throw std::invalid_argument(
            "assemble_stiffness_loop: n_passes must be >= 1 (got "
            + std::to_string(n_passes) + ")");
    }

    // Detect extraordinary interior vertices (interior valence != 6).
    // Boundary vertices are handled separately by the Schweitzer
    // phantom-vertex augmentation and do not require subdivision.
    const auto edges    = build_edges(F);
    const auto valences = vertex_valences(V.rows(), edges);
    std::vector<bool> is_bdry(static_cast<std::size_t>(V.rows()), false);
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            is_bdry[static_cast<std::size_t>(e.v0)] = true;
            is_bdry[static_cast<std::size_t>(e.v1)] = true;
        }
    }
    bool any_extraordinary = false;
    for (Eigen::Index v = 0; v < V.rows(); ++v) {
        if (!is_bdry[static_cast<std::size_t>(v)]
            && valences[static_cast<std::size_t>(v)] != 6)
        {
            any_extraordinary = true;
            break;
        }
    }

    Eigen::SparseMatrix<double> K;
    if (!any_extraordinary) {
        // Fast path: no subdivision needed.
        const auto aug   = augment_for_loop_boundary(V, F);
        const auto K_aug = assemble_stiffness_augmented(
            aug, material,
            /*skip_irregular_triangles=*/false,
            /*use_stam_for_irregular=*/false,
            rule);
        Eigen::SparseMatrix<double> Ct = aug.C.transpose();
        K = Ct * K_aug * aug.C;
    } else {
        // Pre-subdivide so that every face has at most one extraordinary
        // corner. n_passes Loop steps + augment + assemble.
        const auto sub = loop_subdivide_n_times(V, F, n_passes);
        const auto aug = augment_for_loop_boundary(sub.V_sub, sub.F_sub);
        const auto K_sub_aug = assemble_stiffness_augmented(
            aug, material,
            /*skip_irregular_triangles=*/!use_stam,
            /*use_stam_for_irregular=*/use_stam,
            rule);
        Eigen::SparseMatrix<double> Ct    = aug.C.transpose();
        Eigen::SparseMatrix<double> K_sub = Ct * K_sub_aug * aug.C;
        Eigen::SparseMatrix<double> St    = sub.S.transpose();
        K = St * K_sub * sub.S;
    }

    // Belt-and-suspenders finiteness sweep on the assembled stream,
    // matching the LME assembler. The per-quadrature-point a_det guards
    // are the root-cause fix; this O(nnz) check catches any other
    // non-finite source before it reaches the eigensolver as an opaque
    // failure.
    K.makeCompressed();
    if (!K.coeffs().allFinite()) {
        throw std::runtime_error(
            "assemble_stiffness_loop: assembled stiffness matrix "
            "contains a non-finite entry (NaN/Inf) — likely a non-finite "
            "vertex position or degenerate geometry upstream.");
    }
    return K;
}

}  // namespace chladni::shell::loop
