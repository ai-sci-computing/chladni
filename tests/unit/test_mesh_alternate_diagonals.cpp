/**
 * @file test_mesh_alternate_diagonals.cpp
 * @brief Pin the @ref chladni::mesh::QuadSplit schemes of the
 *   structured procedural generators
 *   (@ref chladni::mesh::generate_circular_disk,
 *   @ref chladni::mesh::generate_annulus,
 *   @ref chladni::mesh::generate_cylinder) to their purpose: exactly
 *   MIRROR-SYMMETRIC triangulations.
 *
 * Background (2026-06-03 chirality investigation): the legacy
 * Consistent split cuts every axial/annular quad with the SAME
 * diagonal. That preserves the C_N rotational symmetry (doublet
 * frequencies stay degenerate to machine precision) but breaks every
 * reflection — the mesh has a handedness. With no mirror in the
 * discrete operator the degenerate Chladni doublets have no symmetry
 * axis to pin them, and the displayed nodal patterns come out visibly
 * skewed/chiral. Two cures, both pinned here (face-set
 * mirror-residuals drop from 0.86–1.0 to exactly 0):
 *  - Checkerboard alternates the diagonal by (ring + j) parity at
 *    unchanged vertex count, but leaves rim vertices of alternating
 *    valence 3/5 (a period-2 rim modulation that imprints visible
 *    artefacts of its own);
 *  - UnionJack adds a centre vertex + 4 triangles per quad — full
 *    dihedral symmetry, uniform rims, and the best measured accuracy
 *    (32x8 disk Leissa n=2 on SME: +0.058 % vs Consistent +0.713 %,
 *    and still 7.7× better than a Consistent mesh refined to the same
 *    vertex count).
 * Neither symmetric scheme is Loop- or 1st-order-LME-compatible (rim
 * valence ≠ 4); both are for the 2nd-order SME path (the GUI default).
 * Defaults stay Consistent so existing fixtures are untouched.
 */

#include <chladni/mesh.hpp>

#include <Eigen/Geometry>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

/// Canonical (sorted) vertex triple — face identity independent of winding.
std::array<int, 3> face_key(int a, int b, int c)
{
    std::array<int, 3> k{a, b, c};
    std::sort(k.begin(), k.end());
    return k;
}

/// Fraction of faces whose image under the vertex permutation @p perm is
/// NOT a face of the mesh. 0 = the face set is exactly invariant.
double mirror_residual(const Eigen::MatrixXi& F, const std::vector<int>& perm)
{
    std::set<std::array<int, 3>> faces;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        faces.insert(face_key(F(t, 0), F(t, 1), F(t, 2)));
    }
    Eigen::Index bad = 0;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        if (faces.count(face_key(perm[static_cast<std::size_t>(F(t, 0))],
                                 perm[static_cast<std::size_t>(F(t, 1))],
                                 perm[static_cast<std::size_t>(F(t, 2))]))
            == 0)
        {
            ++bad;
        }
    }
    return static_cast<double>(bad) / static_cast<double>(F.rows());
}

/// Mirror across the x-axis (θ → −θ) for the polar disk vertex layout:
/// centre fixed, ring vertex j → (N − j) mod N within each ring.
std::vector<int> disk_mirror_perm(int n_az, int n_rad)
{
    std::vector<int> p(static_cast<std::size_t>(1 + n_rad * n_az));
    p[0] = 0;
    for (int r = 1; r <= n_rad; ++r) {
        for (int j = 0; j < n_az; ++j) {
            p[static_cast<std::size_t>(1 + (r - 1) * n_az + j)] =
                1 + (r - 1) * n_az + ((n_az - j) % n_az);
        }
    }
    return p;
}

/// Same mirror for the annulus layout (ring-major, no centre vertex).
std::vector<int> annulus_mirror_perm(int n_az, int n_rad)
{
    std::vector<int> p(static_cast<std::size_t>(n_rad * n_az));
    for (int r = 0; r < n_rad; ++r) {
        for (int j = 0; j < n_az; ++j) {
            p[static_cast<std::size_t>(r * n_az + j)] =
                r * n_az + ((n_az - j) % n_az);
        }
    }
    return p;
}

/// Same mirror for the cylinder layout (ring-major along the axis).
std::vector<int> cylinder_mirror_perm(int n_around, int n_along)
{
    std::vector<int> p(
        static_cast<std::size_t>((n_along + 1) * n_around));
    for (int j = 0; j <= n_along; ++j) {
        for (int i = 0; i < n_around; ++i) {
            p[static_cast<std::size_t>(j * n_around + i)] =
                j * n_around + ((n_around - i) % n_around);
        }
    }
    return p;
}

/// Number of boundary edges (edges with exactly one adjacent face).
int count_boundary_edges(const Eigen::MatrixXi& F)
{
    std::set<std::array<int, 2>> seen;
    std::set<std::array<int, 2>> twice;
    for (Eigen::Index t = 0; t < F.rows(); ++t) {
        for (int e = 0; e < 3; ++e) {
            int u = F(t, e), v = F(t, (e + 1) % 3);
            if (u > v) std::swap(u, v);
            const std::array<int, 2> k{u, v};
            if (!seen.insert(k).second) twice.insert(k);
        }
    }
    return static_cast<int>(seen.size() - twice.size());
}

/// All triangles CCW as seen from +z (planar meshes): signed double
/// area > 0 for every face.
bool all_ccw_from_plus_z(const chladni::mesh::TriMesh& m)
{
    for (Eigen::Index t = 0; t < m.F.rows(); ++t) {
        const auto a = m.V.row(m.F(t, 0));
        const auto b = m.V.row(m.F(t, 1));
        const auto c = m.V.row(m.F(t, 2));
        const double cross_z = (b(0) - a(0)) * (c(1) - a(1))
                             - (b(1) - a(1)) * (c(0) - a(0));
        if (!(cross_z > 0.0)) return false;
    }
    return true;
}

/// All face normals point away from the cylinder axis (outward).
bool all_outward_cylinder(const chladni::mesh::TriMesh& m)
{
    for (Eigen::Index t = 0; t < m.F.rows(); ++t) {
        const Eigen::Vector3d a = m.V.row(m.F(t, 0));
        const Eigen::Vector3d b = m.V.row(m.F(t, 1));
        const Eigen::Vector3d c = m.V.row(m.F(t, 2));
        const Eigen::Vector3d n = (b - a).cross(c - a);
        Eigen::Vector3d centroid = (a + b + c) / 3.0;
        centroid(2) = 0.0;  // radial direction at the centroid
        if (!(n.dot(centroid) > 0.0)) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("alternate_diagonals: default off reproduces the legacy face tables",
          "[mesh][procedural][alternate_diagonals]")
{
    // The trailing parameter must not perturb existing meshes: an
    // explicit `false` is byte-identical to the historical default.
    const auto d0 = chladni::mesh::generate_circular_disk(0.1, 16, 3);
    const auto d1 = chladni::mesh::generate_circular_disk(0.1, 16, 3, chladni::mesh::QuadSplit::Consistent);
    REQUIRE((d0.F.array() == d1.F.array()).all());

    const auto a0 = chladni::mesh::generate_annulus(0.1, 0.04, 16, 4);
    const auto a1 = chladni::mesh::generate_annulus(0.1, 0.04, 16, 4, chladni::mesh::QuadSplit::Consistent);
    REQUIRE((a0.F.array() == a1.F.array()).all());

    const auto c0 = chladni::mesh::generate_cylinder(0.1, 0.3, 16, 4);
    const auto c1 = chladni::mesh::generate_cylinder(0.1, 0.3, 16, 4, chladni::mesh::QuadSplit::Consistent);
    REQUIRE((c0.F.array() == c1.F.array()).all());
}

TEST_CASE("alternate_diagonals: identical vertex/face counts and positions",
          "[mesh][procedural][alternate_diagonals]")
{
    const auto d0 = chladni::mesh::generate_circular_disk(0.1, 32, 4);
    const auto d1 = chladni::mesh::generate_circular_disk(0.1, 32, 4, chladni::mesh::QuadSplit::Checkerboard);
    REQUIRE((d0.V.array() == d1.V.array()).all());
    REQUIRE(d0.F.rows() == d1.F.rows());

    const auto a0 = chladni::mesh::generate_annulus(0.1, 0.04, 32, 4);
    const auto a1 = chladni::mesh::generate_annulus(0.1, 0.04, 32, 4, chladni::mesh::QuadSplit::Checkerboard);
    REQUIRE((a0.V.array() == a1.V.array()).all());
    REQUIRE(a0.F.rows() == a1.F.rows());

    const auto c0 = chladni::mesh::generate_cylinder(0.1, 0.3, 32, 8);
    const auto c1 = chladni::mesh::generate_cylinder(0.1, 0.3, 32, 8, chladni::mesh::QuadSplit::Checkerboard);
    REQUIRE((c0.V.array() == c1.V.array()).all());
    REQUIRE(c0.F.rows() == c1.F.rows());
}

TEST_CASE("alternate_diagonals: restores an exact mirror symmetry (even N)",
          "[mesh][procedural][alternate_diagonals]")
{
    // The whole point of the option: the face set becomes exactly
    // invariant under the azimuthal reflection j → (N − j) mod N,
    // while the legacy consistent-diagonal mesh is almost entirely
    // NON-invariant (the chirality the Chladni figures showed).
    {
        const auto perm = disk_mirror_perm(32, 4);
        const auto cons = chladni::mesh::generate_circular_disk(0.1, 32, 4);
        const auto alt  = chladni::mesh::generate_circular_disk(0.1, 32, 4,
        chladni::mesh::QuadSplit::Checkerboard);
        REQUIRE(mirror_residual(cons.F, perm) > 0.5);   // chiral
        REQUIRE(mirror_residual(alt.F,  perm) == 0.0);  // exact mirror
    }
    {
        const auto perm = annulus_mirror_perm(32, 5);
        const auto cons = chladni::mesh::generate_annulus(0.1, 0.04, 32, 5);
        const auto alt  = chladni::mesh::generate_annulus(
            0.1, 0.04, 32, 5, chladni::mesh::QuadSplit::Checkerboard);
        REQUIRE(mirror_residual(cons.F, perm) > 0.5);
        REQUIRE(mirror_residual(alt.F,  perm) == 0.0);
    }
    {
        const auto perm = cylinder_mirror_perm(32, 8);
        const auto cons = chladni::mesh::generate_cylinder(0.1, 0.3, 32, 8);
        const auto alt  = chladni::mesh::generate_cylinder(0.1, 0.3, 32, 8,
        chladni::mesh::QuadSplit::Checkerboard);
        REQUIRE(mirror_residual(cons.F, perm) > 0.5);
        REQUIRE(mirror_residual(alt.F,  perm) == 0.0);
    }
}

TEST_CASE("alternate_diagonals: odd azimuthal count throws",
          "[mesh][procedural][alternate_diagonals]")
{
    // The checkerboard needs an even azimuthal count to close around
    // the wrap; with odd N a seam defect silently re-introduces the
    // chirality (measured residual stays 1.0), so the generators fail
    // loudly instead.
    REQUIRE_THROWS_AS(
        chladni::mesh::generate_circular_disk(0.1, 33, 4, chladni::mesh::QuadSplit::Checkerboard),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        chladni::mesh::generate_annulus(0.1, 0.04, 33, 4, chladni::mesh::QuadSplit::Checkerboard),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        chladni::mesh::generate_cylinder(0.1, 0.3, 33, 4, chladni::mesh::QuadSplit::Checkerboard),
        std::invalid_argument);
    // ...and even N does not throw.
    REQUIRE_NOTHROW(
        chladni::mesh::generate_circular_disk(0.1, 34, 4, chladni::mesh::QuadSplit::Checkerboard));
}

TEST_CASE("symmetric splits: manifold boundaries and winding preserved",
          "[mesh][procedural][alternate_diagonals]")
{
    using chladni::mesh::QuadSplit;
    for (QuadSplit split : {QuadSplit::Checkerboard, QuadSplit::UnionJack}) {
        // Disk: N rim edges, all faces CCW from +z.
        {
            const auto m = chladni::mesh::generate_circular_disk(
                0.1, 32, 4, split);
            REQUIRE(count_boundary_edges(m.F) == 32);
            REQUIRE(all_ccw_from_plus_z(m));
        }
        // Annulus: inner + outer rims = 2N boundary edges, CCW from +z.
        {
            const auto m = chladni::mesh::generate_annulus(
                0.1, 0.04, 32, 5, split);
            REQUIRE(count_boundary_edges(m.F) == 64);
            REQUIRE(all_ccw_from_plus_z(m));
        }
        // Cylinder: two end loops = 2N boundary edges, outward normals.
        {
            const auto m = chladni::mesh::generate_cylinder(
                0.1, 0.3, 32, 8, split);
            REQUIRE(count_boundary_edges(m.F) == 64);
            REQUIRE(all_outward_cylinder(m));
        }
    }
}

TEST_CASE("union-jack: full symmetry, documented counts, odd N allowed",
          "[mesh][procedural][alternate_diagonals][union_jack]")
{
    using chladni::mesh::QuadSplit;
    // Counts: one centre vertex + 4 triangles per strip quad.
    {
        const int N = 32, nr = 8;
        const auto m = chladni::mesh::generate_circular_disk(
            0.1, N, nr, QuadSplit::UnionJack);
        REQUIRE(m.num_vertices() == 1 + (2 * nr - 1) * N);
        REQUIRE(m.num_faces()    == N * (4 * nr - 3));
    }
    {
        const int N = 32, nr = 5;
        const auto m = chladni::mesh::generate_annulus(
            0.1, 0.04, N, nr, QuadSplit::UnionJack);
        REQUIRE(m.num_vertices() == (2 * nr - 1) * N);
        REQUIRE(m.num_faces()    == 4 * N * (nr - 1));
    }
    {
        const int N = 32, nl = 8;
        const auto m = chladni::mesh::generate_cylinder(
            0.1, 0.3, N, nl, QuadSplit::UnionJack);
        REQUIRE(m.num_vertices() == N * (2 * nl + 1));
        REQUIRE(m.num_faces()    == 4 * N * nl);
    }
    // Exact mirror symmetry — extend the ring-vertex mirror with the
    // centre-vertex images (quad j maps to quad N-1-j within each
    // strip; the centre vertices follow).
    {
        const int N = 32, nr = 8;
        const auto m = chladni::mesh::generate_circular_disk(
            0.1, N, nr, QuadSplit::UnionJack);
        std::vector<int> perm = disk_mirror_perm(N, nr);
        const int ctr_base = 1 + nr * N;
        perm.resize(static_cast<std::size_t>(m.num_vertices()));
        for (int r = 1; r < nr; ++r) {
            for (int j = 0; j < N; ++j) {
                perm[static_cast<std::size_t>(
                    ctr_base + (r - 1) * N + j)] =
                    ctr_base + (r - 1) * N + ((N - 1 - j) % N);
            }
        }
        REQUIRE(mirror_residual(m.F, perm) == 0.0);
    }
    // No seam-parity constraint: every quad is treated identically, so
    // odd azimuthal counts are fine (unlike Checkerboard).
    REQUIRE_NOTHROW(chladni::mesh::generate_circular_disk(
        0.1, 33, 4, QuadSplit::UnionJack));
    REQUIRE_NOTHROW(chladni::mesh::generate_cylinder(
        0.1, 0.3, 33, 4, QuadSplit::UnionJack));
}
