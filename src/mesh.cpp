/**
 * @file mesh.cpp
 * @brief Implementation of chladni::mesh::load_obj using tinyobjloader.
 *
 * tinyobjloader is configured to triangulate polygons during parse,
 * so we only ever see triangle faces here. We unpack tinyobj's
 * float-precision vertex array into Eigen::MatrixXd, copy the per-face
 * vertex indices into Eigen::MatrixXi, and validate every index lies in
 * @f$[0, n)@f$ before returning.
 */

#include <chladni/mesh.hpp>

#include <tiny_obj_loader.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <string>

namespace chladni::mesh {

TriMesh load_obj(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("load_obj: file does not exist: "
                                 + path.string());
    }

    tinyobj::ObjReaderConfig config;
    config.triangulate  = true;   // fan-triangulate polygons during parse
    config.vertex_color = false;  // we don't use per-vertex colour

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), config)) {
        std::string msg = "load_obj: failed to parse '" + path.string() + "'";
        if (!reader.Error().empty()) {
            msg += ": " + reader.Error();
        }
        throw std::runtime_error(msg);
    }
    if (!reader.Warning().empty()) {
        // Warnings (e.g. missing material library) are not fatal here.
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    if (attrib.vertices.empty()) {
        throw std::runtime_error("load_obj: no vertices in '"
                                 + path.string() + "'");
    }
    if ((attrib.vertices.size() % 3u) != 0u) {
        throw std::runtime_error("load_obj: vertex array length is not "
                                 "a multiple of 3");
    }

    const auto n_v = attrib.vertices.size() / 3u;

    TriMesh out;
    out.V.resize(static_cast<Eigen::Index>(n_v), 3);
    for (std::size_t i = 0; i < n_v; ++i) {
        const auto row = static_cast<Eigen::Index>(i);
        out.V(row, 0) = static_cast<double>(attrib.vertices[3u * i + 0u]);
        out.V(row, 1) = static_cast<double>(attrib.vertices[3u * i + 1u]);
        out.V(row, 2) = static_cast<double>(attrib.vertices[3u * i + 2u]);
    }

    // Count triangles across all shape groups. tinyobjloader's
    // num_face_vertices element type widened in v2.x, so we just
    // accept whatever integer type it gives us.
    std::size_t n_f = 0;
    for (const auto& s : shapes) {
        for (const auto k : s.mesh.num_face_vertices) {
            if (k != 3) {
                throw std::runtime_error(
                    "load_obj: non-triangle face detected after "
                    "triangulation; OBJ contains malformed faces");
            }
            ++n_f;
        }
    }

    out.F.resize(static_cast<Eigen::Index>(n_f), 3);
    Eigen::Index out_row = 0;
    for (const auto& s : shapes) {
        std::size_t flat_idx = 0;  // index into s.mesh.indices
        for (std::size_t /*face*/ unused = 0; unused < s.mesh.num_face_vertices.size(); ++unused) {
            for (std::size_t j = 0; j < 3u; ++j) {
                const auto v_idx = s.mesh.indices[flat_idx + j].vertex_index;
                if (v_idx < 0 ||
                    static_cast<std::size_t>(v_idx) >= n_v)
                {
                    throw std::runtime_error(
                        "load_obj: face vertex index out of range");
                }
                out.F(out_row, static_cast<Eigen::Index>(j)) = v_idx;
            }
            flat_idx += 3u;
            ++out_row;
        }
    }

    // Reject degenerate faces (two equal indices => zero area).
    for (Eigen::Index i = 0; i < out.F.rows(); ++i) {
        const auto a = out.F(i, 0);
        const auto b = out.F(i, 1);
        const auto c = out.F(i, 2);
        if (a == b || b == c || a == c) {
            throw std::runtime_error(
                "load_obj: degenerate face (repeated vertex index) at row "
                + std::to_string(i));
        }
    }

    return out;
}

TriMesh generate_cylinder(double radius,
                          double length,
                          int n_around,
                          int n_along,
                          QuadSplit split)
{
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "generate_cylinder: radius must be > 0");
    }
    if (length <= 0.0) {
        throw std::invalid_argument(
            "generate_cylinder: length must be > 0");
    }
    if (n_around < 3) {
        throw std::invalid_argument(
            "generate_cylinder: n_around must be >= 3");
    }
    if (n_along < 1) {
        throw std::invalid_argument(
            "generate_cylinder: n_along must be >= 1");
    }
    if (split == QuadSplit::Checkerboard && (n_around % 2 != 0)) {
        throw std::invalid_argument(
            "generate_cylinder: QuadSplit::Checkerboard requires an even "
            "n_around (odd leaves a checkerboard seam defect that "
            "silently re-introduces the mesh chirality); got "
            + std::to_string(n_around));
    }

    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;

    const bool union_jack = (split == QuadSplit::UnionJack);
    const Eigen::Index n_rings = static_cast<Eigen::Index>(n_along) + 1;
    const Eigen::Index n_per_ring = static_cast<Eigen::Index>(n_around);
    // UnionJack adds one centre vertex per quad (n_along * n_around of
    // them, indexed after the ring vertices) and emits 4 triangles per
    // quad instead of 2.
    const Eigen::Index n_quads =
        static_cast<Eigen::Index>(n_along) * n_per_ring;
    const Eigen::Index n_v =
        n_rings * n_per_ring + (union_jack ? n_quads : 0);
    const Eigen::Index n_f = (union_jack ? 4 : 2) * n_quads;

    TriMesh out;
    out.V.resize(n_v, 3);
    out.F.resize(n_f, 3);

    // Lay vertices ring by ring, around-the-ring within each ring.
    // Wrap-around is handled by the (i + 1) % n_around face indexing,
    // not by duplicating the i = 0 vertex — the result is therefore a
    // proper topological tube with no seam.
    for (Eigen::Index j = 0; j < n_rings; ++j) {
        const double z = length * static_cast<double>(j)
                       / static_cast<double>(n_along);
        for (Eigen::Index i = 0; i < n_per_ring; ++i) {
            const double theta = two_pi * static_cast<double>(i)
                               / static_cast<double>(n_around);
            const Eigen::Index row = j * n_per_ring + i;
            out.V(row, 0) = radius * std::cos(theta);
            out.V(row, 1) = radius * std::sin(theta);
            out.V(row, 2) = z;
        }
    }

    // Each axial-circumferential quad between rings j and j+1 is split
    // by the consistent diagonal R_j[i] -> R_{j+1}[i+1], matching the
    // bundled cylinder.obj — or by the (i + j)-parity checkerboard /
    // the union-jack centre-vertex fan that restore the mirror
    // symmetry the consistent split breaks (see the QuadSplit doc).
    // Face winding is chosen so the outward normal points away from
    // the cylinder axis in every variant.
    //
    // UnionJack centre vertices sit ON the mid-surface (parametric
    // midpoint: half-step angle, half-step z), NOT at the chord
    // centroid (which would fall inside the cylinder by the sagitta).
    if (union_jack) {
        const Eigen::Index ctr_base = n_rings * n_per_ring;
        for (Eigen::Index j = 0; j < n_along; ++j) {
            const double z_mid = length * (static_cast<double>(j) + 0.5)
                               / static_cast<double>(n_along);
            for (Eigen::Index i = 0; i < n_per_ring; ++i) {
                const double theta_mid =
                    two_pi * (static_cast<double>(i) + 0.5)
                    / static_cast<double>(n_around);
                const Eigen::Index row = ctr_base + j * n_per_ring + i;
                out.V(row, 0) = radius * std::cos(theta_mid);
                out.V(row, 1) = radius * std::sin(theta_mid);
                out.V(row, 2) = z_mid;
            }
        }
    }
    Eigen::Index f = 0;
    for (Eigen::Index j = 0; j < n_along; ++j) {
        for (Eigen::Index i = 0; i < n_per_ring; ++i) {
            const Eigen::Index i_next = (i + 1) % n_per_ring;
            const Eigen::Index a = j       * n_per_ring + i;
            const Eigen::Index b = j       * n_per_ring + i_next;
            const Eigen::Index c = (j + 1) * n_per_ring + i_next;
            const Eigen::Index d = (j + 1) * n_per_ring + i;
            if (union_jack) {
                // Fan around the on-surface centre m; the quad boundary
                // a→b→c→d is CCW seen from outside, so each (edge, m)
                // triangle below keeps the outward winding.
                const Eigen::Index m =
                    n_rings * n_per_ring + j * n_per_ring + i;
                const std::array<std::array<Eigen::Index, 3>, 4> tris{{
                    {a, b, m}, {b, c, m}, {c, d, m}, {d, a, m}}};
                for (const auto& t : tris) {
                    out.F(f, 0) = static_cast<int>(t[0]);
                    out.F(f, 1) = static_cast<int>(t[1]);
                    out.F(f, 2) = static_cast<int>(t[2]);
                    ++f;
                }
            } else if (!(split == QuadSplit::Checkerboard
                         && ((i + j) % 2 == 1))) {
                // Diagonal a-c.
                // Triangle 1: a, b, c (lower-right of the quad).
                out.F(f, 0) = static_cast<int>(a);
                out.F(f, 1) = static_cast<int>(b);
                out.F(f, 2) = static_cast<int>(c);
                ++f;
                // Triangle 2: a, c, d (upper-left of the quad).
                out.F(f, 0) = static_cast<int>(a);
                out.F(f, 1) = static_cast<int>(c);
                out.F(f, 2) = static_cast<int>(d);
                ++f;
            } else {
                // Diagonal b-d (mirrored split, same outward winding).
                // Triangle 1: b, c, d (upper-right of the quad).
                out.F(f, 0) = static_cast<int>(b);
                out.F(f, 1) = static_cast<int>(c);
                out.F(f, 2) = static_cast<int>(d);
                ++f;
                // Triangle 2: b, d, a (lower-left of the quad).
                out.F(f, 0) = static_cast<int>(b);
                out.F(f, 1) = static_cast<int>(d);
                out.F(f, 2) = static_cast<int>(a);
                ++f;
            }
        }
    }

    return out;
}

TriMesh generate_flat_plate(double length_a,
                            double length_b,
                            int    n_x,
                            int    n_y)
{
    if (length_a <= 0.0) {
        throw std::invalid_argument(
            "generate_flat_plate: length_a must be > 0");
    }
    if (length_b <= 0.0) {
        throw std::invalid_argument(
            "generate_flat_plate: length_b must be > 0");
    }
    if (n_x < 2) {
        throw std::invalid_argument(
            "generate_flat_plate: n_x must be >= 2 "
            "(n_x = 1 would put two chamfered corners on the same edge)");
    }
    if (n_y < 2) {
        throw std::invalid_argument(
            "generate_flat_plate: n_y must be >= 2");
    }

    // (i, j) with 0 <= i <= n_x and 0 <= j <= n_y indexes a candidate
    // grid vertex. The two omitted corners are (n_x, 0) and (0, n_y).
    auto vid_full = [&](int i, int j) -> int {
        return i + j * (n_x + 1);
    };
    const int n_full        = (n_x + 1) * (n_y + 1);
    const int omitted_a     = vid_full(n_x, 0);
    const int omitted_b     = vid_full(0, n_y);
    const Eigen::Index n_v  = static_cast<Eigen::Index>(n_full - 2);

    // Build the dense remap from full-grid index to compact vertex ID,
    // skipping the 2 omitted corners. Walk the grid in row-major order
    // (j outer, i inner) so the resulting vertex table is laid out
    // bottom-row-first.
    std::vector<int> remap(static_cast<std::size_t>(n_full), -1);
    int next_id = 0;
    for (int j = 0; j <= n_y; ++j) {
        for (int i = 0; i <= n_x; ++i) {
            const int vf = vid_full(i, j);
            if (vf == omitted_a || vf == omitted_b) continue;
            remap[static_cast<std::size_t>(vf)] = next_id++;
        }
    }
    auto vid = [&](int i, int j) -> int {
        return remap[static_cast<std::size_t>(vid_full(i, j))];
    };

    TriMesh out;
    out.V.resize(n_v, 3);
    for (int j = 0; j <= n_y; ++j) {
        for (int i = 0; i <= n_x; ++i) {
            const int vf = vid_full(i, j);
            const int id = remap[static_cast<std::size_t>(vf)];
            if (id < 0) continue;
            const double x = length_a
                * static_cast<double>(i) / static_cast<double>(n_x);
            const double y = length_b
                * static_cast<double>(j) / static_cast<double>(n_y);
            out.V(id, 0) = x;
            out.V(id, 1) = y;
            out.V(id, 2) = 0.0;
        }
    }

    // Triangulate cell by cell with the up-right diagonal
    //   T1 = (i,   j  ) -> (i+1, j  ) -> (i+1, j+1)   "lower right"
    //   T2 = (i,   j  ) -> (i+1, j+1) -> (i,   j+1)   "upper left"
    // Skip T1 in the cell that holds the omitted (n_x, 0) corner
    // (cell (n_x-1, 0)) and skip T2 in the cell that holds the omitted
    // (0, n_y) corner (cell (0, n_y-1)).
    const Eigen::Index n_f = 2 * n_x * n_y - 2;
    out.F.resize(n_f, 3);
    Eigen::Index f = 0;
    for (int j = 0; j < n_y; ++j) {
        for (int i = 0; i < n_x; ++i) {
            const bool t1_holds_omitted = (i == n_x - 1 && j == 0);
            const bool t2_holds_omitted = (i == 0 && j == n_y - 1);

            if (!t1_holds_omitted) {
                out.F(f, 0) = vid(i,     j    );
                out.F(f, 1) = vid(i + 1, j    );
                out.F(f, 2) = vid(i + 1, j + 1);
                ++f;
            }
            if (!t2_holds_omitted) {
                out.F(f, 0) = vid(i,     j    );
                out.F(f, 1) = vid(i + 1, j + 1);
                out.F(f, 2) = vid(i,     j + 1);
                ++f;
            }
        }
    }
    // Sanity check the count — should match the analytic formula above.
    if (f != n_f) {
        throw std::runtime_error(
            "generate_flat_plate: internal triangle-count mismatch (got "
            + std::to_string(static_cast<long long>(f)) + ", expected "
            + std::to_string(static_cast<long long>(n_f)) + ")");
    }

    return out;
}

TriMesh generate_icosphere(double radius, int n_subdivisions)
{
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "generate_icosphere: radius must be > 0");
    }
    if (n_subdivisions < 0) {
        throw std::invalid_argument(
            "generate_icosphere: n_subdivisions must be >= 0 (got "
            + std::to_string(n_subdivisions) + ")");
    }

    // Standard golden-ratio icosahedron, vertex layout from
    // three.js's IcosahedronGeometry. Vertices at distance
    // sqrt(1 + t^2) from origin (t = (1+sqrt(5))/2 is the golden
    // ratio); we normalise to the unit sphere first and scale to
    // radius at the end.
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    Eigen::MatrixXd V(12, 3);
    V << -1,  t,  0,
          1,  t,  0,
         -1, -t,  0,
          1, -t,  0,
          0, -1,  t,
          0,  1,  t,
          0, -1, -t,
          0,  1, -t,
          t,  0, -1,
          t,  0,  1,
         -t,  0, -1,
         -t,  0,  1;
    Eigen::MatrixXi F(20, 3);
    F <<  0, 11,  5,
          0,  5,  1,
          0,  1,  7,
          0,  7, 10,
          0, 10, 11,
          1,  5,  9,
          5, 11,  4,
         11, 10,  2,
         10,  7,  6,
          7,  1,  8,
          3,  9,  4,
          3,  4,  2,
          3,  2,  6,
          3,  6,  8,
          3,  8,  9,
          4,  9,  5,
          2,  4, 11,
          6,  2, 10,
          8,  6,  7,
          9,  8,  1;

    auto project_unit = [](Eigen::Matrix<double, 1, 3>& row) {
        const double n = row.norm();
        if (n > 0.0) row /= n;
    };
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        Eigen::Matrix<double, 1, 3> r = V.row(i);
        project_unit(r);
        V.row(i) = r;
    }

    // Subdivide n_subdivisions times. Each round splits every edge
    // at its midpoint, projects the new vertex onto the unit sphere,
    // and replaces every parent face with 4 sub-faces in the standard
    // Loop layout: 3 corner sub-faces + 1 central medial sub-face.
    for (int s = 0; s < n_subdivisions; ++s) {
        // Map each ordered edge (min, max) -> new midpoint vertex
        // index, deduplicating shared edges between adjacent faces.
        std::map<std::pair<int, int>, int> midpoint;
        std::vector<Eigen::Matrix<double, 1, 3>> new_vertices;

        auto get_midpoint = [&](int a, int b) -> int {
            const auto key = std::minmax(a, b);
            const auto it = midpoint.find({key.first, key.second});
            if (it != midpoint.end()) return it->second;
            Eigen::Matrix<double, 1, 3> mid =
                0.5 * (V.row(a) + V.row(b));
            project_unit(mid);
            const int new_idx =
                static_cast<int>(V.rows())
                + static_cast<int>(new_vertices.size());
            new_vertices.push_back(mid);
            midpoint[{key.first, key.second}] = new_idx;
            return new_idx;
        };

        Eigen::MatrixXi F_new(4 * F.rows(), 3);
        for (Eigen::Index f = 0; f < F.rows(); ++f) {
            const int a = F(f, 0);
            const int b = F(f, 1);
            const int c = F(f, 2);
            const int e_ab = get_midpoint(a, b);
            const int e_bc = get_midpoint(b, c);
            const int e_ca = get_midpoint(c, a);
            F_new.row(4 * f + 0) << a,    e_ab, e_ca;
            F_new.row(4 * f + 1) << e_ab, b,    e_bc;
            F_new.row(4 * f + 2) << e_ca, e_bc, c;
            F_new.row(4 * f + 3) << e_ab, e_bc, e_ca;
        }

        Eigen::MatrixXd V_new(V.rows()
                              + static_cast<Eigen::Index>(new_vertices.size()),
                              3);
        V_new.topRows(V.rows()) = V;
        for (Eigen::Index i = 0;
             i < static_cast<Eigen::Index>(new_vertices.size()); ++i)
        {
            V_new.row(V.rows() + i) =
                new_vertices[static_cast<std::size_t>(i)];
        }
        V = std::move(V_new);
        F = std::move(F_new);
    }

    V *= radius;

    TriMesh out;
    out.V = std::move(V);
    out.F = std::move(F);
    return out;
}

TriMesh generate_circular_disk(double radius,
                               int    n_azimuthal,
                               int    n_radial,
                               QuadSplit split)
{
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "generate_circular_disk: radius must be > 0");
    }
    if (n_azimuthal < 3) {
        throw std::invalid_argument(
            "generate_circular_disk: n_azimuthal must be >= 3 (got "
            + std::to_string(n_azimuthal) + ")");
    }
    if (n_radial < 1) {
        throw std::invalid_argument(
            "generate_circular_disk: n_radial must be >= 1 (got "
            + std::to_string(n_radial) + ")");
    }
    if (split == QuadSplit::Checkerboard && (n_azimuthal % 2 != 0)) {
        throw std::invalid_argument(
            "generate_circular_disk: QuadSplit::Checkerboard requires "
            "an even n_azimuthal (odd leaves a checkerboard seam defect "
            "that silently re-introduces the mesh chirality); got "
            + std::to_string(n_azimuthal));
    }

    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    const bool union_jack = (split == QuadSplit::UnionJack);
    // UnionJack adds one centre vertex per strip quad ((n_radial-1) *
    // n_azimuthal of them, indexed after the ring vertices) and emits
    // 4 triangles per quad instead of 2. The central fan is unchanged
    // (it is mirror-invariant as-is).
    const Eigen::Index n_quads =
        static_cast<Eigen::Index>(n_radial - 1) * n_azimuthal;
    const Eigen::Index n_v =
        1 + static_cast<Eigen::Index>(n_radial) * n_azimuthal
          + (union_jack ? n_quads : 0);
    const Eigen::Index n_f =
        n_azimuthal + (union_jack ? 4 : 2) * n_quads;

    Eigen::MatrixXd V(n_v, 3);
    Eigen::MatrixXi F(n_f, 3);

    // Vertex 0 at origin. Then rings r = 1..n_radial, each with
    // n_azimuthal vertices at angles 0, 2π/N, 4π/N, ... Index of
    // ring-r vertex at angle index j is: 1 + (r-1)*N + j.
    V.row(0).setZero();
    for (int r = 1; r <= n_radial; ++r) {
        const double rr = static_cast<double>(r)
                        / static_cast<double>(n_radial) * radius;
        for (int j = 0; j < n_azimuthal; ++j) {
            const double theta =
                two_pi * static_cast<double>(j)
                / static_cast<double>(n_azimuthal);
            const Eigen::Index idx =
                1 + static_cast<Eigen::Index>(r - 1) * n_azimuthal + j;
            V(idx, 0) = rr * std::cos(theta);
            V(idx, 1) = rr * std::sin(theta);
            V(idx, 2) = 0.0;
        }
    }

    auto ring_idx = [&](int r, int j) -> Eigen::Index {
        // r in [1, n_radial], j in [0, n_azimuthal) (mod n_azimuthal).
        return 1 + static_cast<Eigen::Index>(r - 1) * n_azimuthal
                 + (j % n_azimuthal);
    };

    Eigen::Index f = 0;
    // Central fan: triangles (0, ring1[j], ring1[j+1]) for j=0..N-1.
    // CCW as seen from +z (angles increase CCW).
    for (int j = 0; j < n_azimuthal; ++j) {
        F(f, 0) = 0;
        F(f, 1) = static_cast<int>(ring_idx(1, j));
        F(f, 2) = static_cast<int>(ring_idx(1, j + 1));
        ++f;
    }
    // Annular strips between ring r and ring r+1, for r=1..n_radial-1.
    // Each "quad" with corners a=ring_r[j], b=ring_r[j+1],
    // c=ring_{r+1}[j+1], d=ring_{r+1}[j] is split by the diagonal b-d
    // into triangles (b, a, d) and (b, d, c), each CCW from +z — or by
    // the (r + j)-parity checkerboard (diagonal a-c on odd parity) /
    // the union-jack centre-vertex fan that restore the mirror
    // symmetry the consistent split breaks (see the QuadSplit doc; the
    // central fan is mirror-invariant as-is). The inner edge (a, b)
    // is traversed b→a in the strips (opposite to the central fan,
    // which uses a→b via triangle (0, a, b)), so the mesh is manifold
    // across the ring-1 boundary in every variant.
    //
    // UnionJack centre vertices sit at the parametric quad midpoint
    // (half-step angle, mid radius). Note the quad boundary a→b→c→d is
    // CW from +z (b→a is the CCW direction along the inner edge), so
    // the CCW fan below walks the boundary as b→a→d→c→b.
    if (union_jack) {
        const Eigen::Index ctr_base =
            1 + static_cast<Eigen::Index>(n_radial) * n_azimuthal;
        for (int r = 1; r < n_radial; ++r) {
            const double rr_mid = (static_cast<double>(r) + 0.5)
                                / static_cast<double>(n_radial) * radius;
            for (int j = 0; j < n_azimuthal; ++j) {
                const double theta_mid =
                    two_pi * (static_cast<double>(j) + 0.5)
                    / static_cast<double>(n_azimuthal);
                const Eigen::Index row =
                    ctr_base
                    + static_cast<Eigen::Index>(r - 1) * n_azimuthal + j;
                V(row, 0) = rr_mid * std::cos(theta_mid);
                V(row, 1) = rr_mid * std::sin(theta_mid);
                V(row, 2) = 0.0;
            }
        }
    }
    for (int r = 1; r < n_radial; ++r) {
        for (int j = 0; j < n_azimuthal; ++j) {
            const int a = static_cast<int>(ring_idx(r,     j));
            const int b = static_cast<int>(ring_idx(r,     j + 1));
            const int c = static_cast<int>(ring_idx(r + 1, j + 1));
            const int d = static_cast<int>(ring_idx(r + 1, j));
            if (union_jack) {
                const int m = static_cast<int>(
                    1 + static_cast<Eigen::Index>(n_radial) * n_azimuthal
                      + static_cast<Eigen::Index>(r - 1) * n_azimuthal
                      + j);
                const std::array<std::array<int, 3>, 4> tris{{
                    {b, a, m}, {a, d, m}, {d, c, m}, {c, b, m}}};
                for (const auto& t : tris) {
                    F(f, 0) = t[0];
                    F(f, 1) = t[1];
                    F(f, 2) = t[2];
                    ++f;
                }
            } else if (!(split == QuadSplit::Checkerboard
                         && ((r + j) % 2 == 1))) {
                // Diagonal b-d.
                F(f, 0) = b;
                F(f, 1) = a;
                F(f, 2) = d;
                ++f;
                F(f, 0) = b;
                F(f, 1) = d;
                F(f, 2) = c;
                ++f;
            } else {
                // Diagonal a-c (mirrored split, still CCW from +z).
                F(f, 0) = a;
                F(f, 1) = d;
                F(f, 2) = c;
                ++f;
                F(f, 0) = a;
                F(f, 1) = c;
                F(f, 2) = b;
                ++f;
            }
        }
    }

    TriMesh out;
    out.V = std::move(V);
    out.F = std::move(F);
    return out;
}

TriMesh generate_disk_iso(double radius, int n_boundary)
{
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "generate_disk_iso: radius must be > 0");
    }
    if (n_boundary < 3) {
        throw std::invalid_argument(
            "generate_disk_iso: n_boundary must be >= 3 (got "
            + std::to_string(n_boundary) + ")");
    }

    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    constexpr double half_sqrt3 = 0.86602540378443864676;  // sqrt(3)/2

    // Edge length of the regular n_boundary-gon inscribed in r=R.
    // This sets the target interior edge length: rings are spaced
    // s * sqrt(3)/2 apart radially and carry round(2*pi*r/s) vertices.
    const double s = 2.0 * radius * std::sin(std::numbers::pi_v<double>
                                             / static_cast<double>(n_boundary));
    const double dr = s * half_sqrt3;

    // Build the rings outside-to-inside so the rim ends up at indices
    // [0, n_boundary). Each ring carries an azimuthal phase: rim at 0
    // (the regular n_boundary-gon, vertex 0 on +x axis), and every
    // inner ring rotated by half its outer neighbour's angular step.
    // The phase accumulates ring-by-ring so successive rings interleave
    // hex-lattice-style — without it, vertex 0 of every ring lines up
    // along +x, producing a visible radial seam at the wrap-around step
    // of each strip.
    struct Ring { Eigen::Index start; int count; double r; double phase; };
    std::vector<Ring> rings;
    rings.reserve(16);

    int   total_v = n_boundary;
    rings.push_back({0, n_boundary, radius, 0.0});

    // First inner ring is forced to N = n_boundary so the rim strip is
    // a clean polar grid — every rim vertex valence 4, no valence-3 fan
    // defects. The triangles in this strip are ~11% non-equilateral
    // (azimuthal_inner ≈ 0.89 s) which is an acceptable price for the
    // boundary regularity. From the second inner ring inward the vertex
    // count adapts (round(2π r / s)).
    double r = radius - dr;
    double phase = 0.0;
    if (r > 0.5 * s) {
        phase += std::numbers::pi_v<double>
               / static_cast<double>(n_boundary);
        rings.push_back({total_v, n_boundary, r, phase});
        total_v += n_boundary;
        r -= dr;
    }

    // Walk inward from the second inner ring onward — adaptive vertex
    // count. Stop when the next ring would land within s/2 of the
    // origin; that's where we close with a single centre vertex.
    while (r > 0.5 * s) {
        // Half-step phase offset accumulates so each ring interleaves
        // hex-lattice-style with its outer neighbour, breaking the
        // radial seam at the strip wrap-around.
        phase += std::numbers::pi_v<double>
               / static_cast<double>(rings.back().count);
        const double n_az_f =
            std::round(two_pi * r / s);
        const int n_az = std::max(3, static_cast<int>(n_az_f));
        rings.push_back({total_v, n_az, r, phase});
        total_v += n_az;
        r -= dr;
    }

    // Centre vertex. There's always exactly one (we stop *before*
    // creating a ring that would collide with the centre).
    const Eigen::Index centre_idx = total_v;
    ++total_v;

    // Allocate V. Faces are accumulated dynamically; pre-counting is
    // possible but adds little (each strip emits N_outer + N_inner
    // triangles, plus the central fan emits N_innermost).
    Eigen::MatrixXd V(total_v, 3);
    V.setZero();

    // Emit ring vertices CCW from angle ring.phase. Rim is at phase 0
    // (vertex 0 on +x axis, matching generate_circular_disk); each
    // inner ring is rotated by the accumulated half-step offset.
    for (const auto& ring : rings) {
        for (int j = 0; j < ring.count; ++j) {
            const double theta = two_pi
                               * static_cast<double>(j)
                               / static_cast<double>(ring.count)
                               + ring.phase;
            const Eigen::Index idx =
                ring.start + static_cast<Eigen::Index>(j);
            V(idx, 0) = ring.r * std::cos(theta);
            V(idx, 1) = ring.r * std::sin(theta);
            V(idx, 2) = 0.0;
        }
    }
    V(centre_idx, 0) = 0.0;
    V(centre_idx, 1) = 0.0;
    V(centre_idx, 2) = 0.0;

    // Triangulate strips. rings[m] is the outer ring of strip m
    // (towards the rim) and rings[m+1] is the inner ring (towards the
    // centre). Walk j = 0..N_outer-1 along the outer ring; at each
    // step, the inner pointer k(j) = floor(j * N_inner / N_outer) is
    // monotonically non-decreasing (modulo wrap). Emit:
    //   - one outer-apex triangle (k(j), outer[j], outer[j+1])
    //   - if k(j+1) > k(j), one inner-apex triangle
    //         (k(j), k(j+1), outer[j+1])
    // The wrap-around step (j = N_outer-1) advances k from k(N_outer-1)
    // back to N_inner (i.e. 0 mod N_inner), closing the strip cleanly.
    // CCW from +z: outer vertices in CCW order from the inner apex give
    // a CCW triangle, since outer ring runs CCW and is "above" inner.
    std::vector<Eigen::RowVector3i> faces;
    faces.reserve(static_cast<std::size_t>(total_v) * 6);

    auto outer_vert = [&](const Ring& ro, int j) -> int {
        return static_cast<int>(ro.start
            + static_cast<Eigen::Index>(j % ro.count));
    };
    auto inner_vert = [&](const Ring& ri, int k) -> int {
        return static_cast<int>(ri.start
            + static_cast<Eigen::Index>(k % ri.count));
    };

    for (std::size_t m = 0; m + 1 < rings.size(); ++m) {
        const Ring& ro = rings[m];
        const Ring& ri = rings[m + 1];
        const int   N_o = ro.count;
        const int   N_i = ri.count;
        for (int j = 0; j < N_o; ++j) {
            const int k_cur = (j     * N_i) / N_o;
            const int k_nxt = ((j+1) * N_i) / N_o;
            // outer-apex: inner k_cur paired with outer (j, j+1).
            faces.push_back({inner_vert(ri, k_cur),
                             outer_vert(ro, j),
                             outer_vert(ro, j + 1)});
            // inner-apex when the inner pointer advanced. Listed in
            // CCW-from-+z order: (inner[adv], outer[j+1], inner[adv+1]).
            // Going CCW around the origin on the inner ring puts the
            // origin on your left and larger-r on your right, so the
            // outer vertex has to sit *between* the two inner vertices
            // in winding order, not after them, for the triangle to
            // come out CCW. At j=N_o-1 the next pointer is N_i (one
            // full revolution), which mod N_i is 0 — the wrap-around
            // closes the strip cleanly.
            if (k_nxt > k_cur) {
                for (int adv = k_cur; adv < k_nxt; ++adv) {
                    faces.push_back({inner_vert(ri, adv),
                                     outer_vert(ro, j + 1),
                                     inner_vert(ri, adv + 1)});
                }
            }
        }
    }

    // Central fan: the innermost ring closes to the centre vertex with
    // N_innermost triangles. CCW from +z: (centre, inner[j], inner[j+1]).
    const Ring& innermost = rings.back();
    for (int j = 0; j < innermost.count; ++j) {
        faces.push_back({static_cast<int>(centre_idx),
                         inner_vert(innermost, j),
                         inner_vert(innermost, j + 1)});
    }

    Eigen::MatrixXi F(static_cast<Eigen::Index>(faces.size()), 3);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        F.row(static_cast<Eigen::Index>(i)) = faces[i];
    }

    TriMesh out;
    out.V = std::move(V);
    out.F = std::move(F);
    return out;
}

TriMesh generate_disk_hex(double radius, int n_layers)
{
    if (radius <= 0.0) {
        throw std::invalid_argument(
            "generate_disk_hex: radius must be > 0");
    }
    if (n_layers < 2) {
        // L = 1 is degenerate: all six layer-1 lattice points ARE the hex
        // corners, which the construction chamfers away, leaving just the
        // centre vertex and zero triangles. L >= 2 is the first non-empty
        // chamfered hex disk.
        throw std::invalid_argument(
            "generate_disk_hex: n_layers must be >= 2 (got "
            + std::to_string(n_layers)
            + "); L=1 chamfers to a single degenerate vertex)");
    }

    const int    L = n_layers;
    const double s = radius / static_cast<double>(L);
    constexpr double sqrt3_over_2 = 0.86602540378443864676;

    // Hex-axial coordinates (a, b). The lattice point's Euclidean position
    // is a * e1 + b * e2 with e1 = (1, 0), e2 = (1/2, sqrt(3)/2). The hex
    // layer (a, b) belongs to is max(|a|, |b|, |a + b|).
    auto hex_layer = [](int a, int b) {
        const int aa = std::abs(a);
        const int bb = std::abs(b);
        const int cc = std::abs(a + b);
        return std::max({aa, bb, cc});
    };

    // Identify the 6 hexagon corner lattice points. We exclude these
    // from the vertex enumeration — they're the only valence-3 boundary
    // vertices in the unchamfered hex disk, and the Schweitzer
    // augmentation handles them but with a less-rich local basis than
    // valence 4 elsewhere on the rim. Replacing each corner with a
    // single "bridge" triangle (across the chord between its two rim
    // neighbours) chamfers the corner off and leaves a clean
    // valence-4 boundary throughout. The price is 6 new valence-5
    // *interior* vertices at the layer-(L-1) corners (each loses one
    // hex-lattice neighbour to the deletion) — Loop / Stam handle
    // interior valence-5 cleanly, so this is a strict topology win.
    const std::array<std::pair<int, int>, 6> hex_corners = {{
        {L, 0}, {0, L}, {-L, L}, {-L, 0}, {0, -L}, {L, -L}
    }};
    auto is_dropped_corner = [&](int a, int b) {
        for (const auto& [ca, cb] : hex_corners) {
            if (a == ca && b == cb) return true;
        }
        return false;
    };

    // Enumerate lattice vertices inside layer <= L (minus dropped corners),
    // build (a, b) -> idx.
    std::vector<std::pair<int, int>> verts;
    verts.reserve(static_cast<std::size_t>(1 + 3 * L * (L + 1)));
    std::map<std::pair<int, int>, int> idx_map;
    for (int a = -L; a <= L; ++a) {
        for (int b = -L; b <= L; ++b) {
            if (hex_layer(a, b) > L) continue;
            if (is_dropped_corner(a, b)) continue;
            idx_map[{a, b}] = static_cast<int>(verts.size());
            verts.emplace_back(a, b);
        }
    }

    Eigen::MatrixXd V(static_cast<Eigen::Index>(verts.size()), 3);
    V.setZero();
    for (std::size_t i = 0; i < verts.size(); ++i) {
        const auto [a, b] = verts[i];
        V(static_cast<Eigen::Index>(i), 0) =
            (static_cast<double>(a) + 0.5 * static_cast<double>(b)) * s;
        V(static_cast<Eigen::Index>(i), 1) =
            sqrt3_over_2 * static_cast<double>(b) * s;
    }

    // Snap the outermost lattice ring (layer L) radially onto r = R so
    // the silhouette is a circle, not a hexagon. Only the rim vertices
    // are touched — interior topology stays a pure hex lattice
    // (valence 6 everywhere except the 6 rim corners, by construction).
    // Edge midpoints on the hexagon were at r = R·sqrt(3)/2 ≈ 0.866·R
    // and move outward by ~15% to land on r = R; corner vertices were
    // already at r = R and don't move. The outermost ring of triangles
    // gets locally distorted but stays well-conditioned (no slivers or
    // inversions).
    for (std::size_t i = 0; i < verts.size(); ++i) {
        const auto [a, b] = verts[i];
        if (hex_layer(a, b) != L) continue;
        const double x = V(static_cast<Eigen::Index>(i), 0);
        const double y = V(static_cast<Eigen::Index>(i), 1);
        const double r = std::sqrt(x * x + y * y);
        if (r > 0.0) {
            const double scale = radius / r;
            V(static_cast<Eigen::Index>(i), 0) = x * scale;
            V(static_cast<Eigen::Index>(i), 1) = y * scale;
        }
    }

    // Emit triangles. Each (a, b) contributes up to two triangles —
    // the "up" rhombus-half (a,b)->(a+1,b)->(a,b+1) and the "down"
    // rhombus-half (a,b)->(a+1,b-1)->(a+1,b). Both are CCW from +z in
    // hex-axial coordinates (verified by computing the cross product
    // with the e1/e2 basis). A triangle is emitted only if all three
    // corners are in-disk — boundary vertices on the rim emit nothing
    // and their incident triangles come from their interior neighbours.
    std::vector<Eigen::RowVector3i> faces;
    faces.reserve(static_cast<std::size_t>(6 * L * L));
    for (const auto& [a, b] : verts) {
        const int i0 = idx_map.at({a, b});

        // Up triangle.
        auto it_e1  = idx_map.find({a + 1, b});
        auto it_e2  = idx_map.find({a, b + 1});
        if (it_e1 != idx_map.end() && it_e2 != idx_map.end()) {
            faces.push_back(Eigen::RowVector3i{
                i0, it_e1->second, it_e2->second});
        }

        // Down triangle.
        auto it_dm = idx_map.find({a + 1, b - 1});
        auto it_d  = idx_map.find({a + 1, b});
        if (it_dm != idx_map.end() && it_d != idx_map.end()) {
            faces.push_back(Eigen::RowVector3i{
                i0, it_dm->second, it_d->second});
        }
    }

    // Six "bridge" triangles closing the gaps left by the dropped
    // hexagon corners. Each connects the two rim neighbours of a
    // dropped corner via its sole interior neighbour. Listed in
    // (rim_ccw, interior, rim_cw) order — CCW from +z for all six
    // (derived by 60° rotation symmetry of the hex lattice, with the
    // (L, 0) and (0, L) cases verified explicitly).
    struct CornerBridge {
        std::pair<int, int> rim_ccw;
        std::pair<int, int> interior;
        std::pair<int, int> rim_cw;
    };
    const std::array<CornerBridge, 6> bridges = {{
        {{L - 1,   1},     {L - 1,   0},     {L,      -1   }}, // corner ( L,  0)
        {{-1,      L},     {0,       L - 1}, {1,       L - 1}}, // corner ( 0,  L)
        {{-L,      L - 1}, {-L + 1,  L - 1}, {-L + 1,  L   }}, // corner (-L,  L)
        {{-L + 1, -1},     {-L + 1,  0},     {-L,      1   }}, // corner (-L,  0)
        {{1,     -L},      {0,      -L + 1}, {-1,     -L + 1}}, // corner ( 0, -L)
        {{L,     -L + 1},  {L - 1,  -L + 1}, {L - 1,  -L   }}, // corner ( L, -L)
    }};
    for (const auto& br : bridges) {
        auto it_a = idx_map.find(br.rim_ccw);
        auto it_i = idx_map.find(br.interior);
        auto it_b = idx_map.find(br.rim_cw);
        if (it_a != idx_map.end() && it_i != idx_map.end()
            && it_b != idx_map.end())
        {
            faces.push_back(Eigen::RowVector3i{
                it_a->second, it_i->second, it_b->second});
        }
    }

    Eigen::MatrixXi F(static_cast<Eigen::Index>(faces.size()), 3);
    for (std::size_t i = 0; i < faces.size(); ++i) {
        F.row(static_cast<Eigen::Index>(i)) = faces[i];
    }

    TriMesh out;
    out.V = std::move(V);
    out.F = std::move(F);
    return out;
}

TriMesh generate_annulus(double radius_outer,
                         double radius_inner,
                         int    n_azimuthal,
                         int    n_radial,
                         QuadSplit split)
{
    if (radius_outer <= 0.0) {
        throw std::invalid_argument(
            "generate_annulus: radius_outer must be > 0");
    }
    if (!(radius_inner > 0.0 && radius_inner < radius_outer)) {
        throw std::invalid_argument(
            "generate_annulus: radius_inner must be in (0, radius_outer)");
    }
    if (n_azimuthal < 3) {
        throw std::invalid_argument(
            "generate_annulus: n_azimuthal must be >= 3 (got "
            + std::to_string(n_azimuthal) + ")");
    }
    if (n_radial < 2) {
        throw std::invalid_argument(
            "generate_annulus: n_radial must be >= 2 (got "
            + std::to_string(n_radial) + ")");
    }
    if (split == QuadSplit::Checkerboard && (n_azimuthal % 2 != 0)) {
        throw std::invalid_argument(
            "generate_annulus: QuadSplit::Checkerboard requires an even "
            "n_azimuthal (odd leaves a checkerboard seam defect that "
            "silently re-introduces the mesh chirality); got "
            + std::to_string(n_azimuthal));
    }

    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    const bool union_jack = (split == QuadSplit::UnionJack);
    const Eigen::Index n_quads =
        static_cast<Eigen::Index>(n_azimuthal) * (n_radial - 1);
    const Eigen::Index n_v =
        static_cast<Eigen::Index>(n_radial) * n_azimuthal
        + (union_jack ? n_quads : 0);
    const Eigen::Index n_f = (union_jack ? 4 : 2) * n_quads;

    Eigen::MatrixXd V(n_v, 3);
    Eigen::MatrixXi F(n_f, 3);

    // Ring r in [0, n_radial-1]. radius interpolates linearly between
    // radius_inner (r=0) and radius_outer (r=n_radial-1). Vertex index
    // for (r, j) is r*n_azimuthal + j.
    for (int r = 0; r < n_radial; ++r) {
        const double t = static_cast<double>(r)
                       / static_cast<double>(n_radial - 1);
        const double rr =
            (1.0 - t) * radius_inner + t * radius_outer;
        for (int j = 0; j < n_azimuthal; ++j) {
            const double theta =
                two_pi * static_cast<double>(j)
                / static_cast<double>(n_azimuthal);
            const Eigen::Index idx =
                static_cast<Eigen::Index>(r) * n_azimuthal + j;
            V(idx, 0) = rr * std::cos(theta);
            V(idx, 1) = rr * std::sin(theta);
            V(idx, 2) = 0.0;
        }
    }

    auto ring_idx = [&](int r, int j) -> Eigen::Index {
        return static_cast<Eigen::Index>(r) * n_azimuthal
             + (j % n_azimuthal);
    };

    // Annular strips: same triangulation as generate_circular_disk's
    // strips (split each quad by the b-d diagonal, two CCW triangles)
    // — or the (r + j)-parity checkerboard / union-jack centre-vertex
    // fan (restore the mirror symmetry; see the QuadSplit doc). The
    // union-jack fan winds b→a→d→c→b for the same reason as the disk
    // (the quad boundary a→b→c→d is CW from +z), with centres at the
    // parametric quad midpoint.
    if (union_jack) {
        const Eigen::Index ctr_base =
            static_cast<Eigen::Index>(n_radial) * n_azimuthal;
        for (int r = 0; r < n_radial - 1; ++r) {
            const double t_mid = (static_cast<double>(r) + 0.5)
                               / static_cast<double>(n_radial - 1);
            const double rr_mid =
                (1.0 - t_mid) * radius_inner + t_mid * radius_outer;
            for (int j = 0; j < n_azimuthal; ++j) {
                const double theta_mid =
                    two_pi * (static_cast<double>(j) + 0.5)
                    / static_cast<double>(n_azimuthal);
                const Eigen::Index row =
                    ctr_base
                    + static_cast<Eigen::Index>(r) * n_azimuthal + j;
                V(row, 0) = rr_mid * std::cos(theta_mid);
                V(row, 1) = rr_mid * std::sin(theta_mid);
                V(row, 2) = 0.0;
            }
        }
    }
    Eigen::Index f = 0;
    for (int r = 0; r < n_radial - 1; ++r) {
        for (int j = 0; j < n_azimuthal; ++j) {
            const int a = static_cast<int>(ring_idx(r,     j));
            const int b = static_cast<int>(ring_idx(r,     j + 1));
            const int c = static_cast<int>(ring_idx(r + 1, j + 1));
            const int d = static_cast<int>(ring_idx(r + 1, j));
            if (union_jack) {
                const int m = static_cast<int>(
                    static_cast<Eigen::Index>(n_radial) * n_azimuthal
                    + static_cast<Eigen::Index>(r) * n_azimuthal + j);
                const std::array<std::array<int, 3>, 4> tris{{
                    {b, a, m}, {a, d, m}, {d, c, m}, {c, b, m}}};
                for (const auto& t : tris) {
                    F(f, 0) = t[0];
                    F(f, 1) = t[1];
                    F(f, 2) = t[2];
                    ++f;
                }
            } else if (!(split == QuadSplit::Checkerboard
                         && ((r + j) % 2 == 1))) {
                // Diagonal b-d.
                F(f, 0) = b;
                F(f, 1) = a;
                F(f, 2) = d;
                ++f;
                F(f, 0) = b;
                F(f, 1) = d;
                F(f, 2) = c;
                ++f;
            } else {
                // Diagonal a-c (mirrored split, still CCW from +z).
                F(f, 0) = a;
                F(f, 1) = d;
                F(f, 2) = c;
                ++f;
                F(f, 0) = a;
                F(f, 1) = c;
                F(f, 2) = b;
                ++f;
            }
        }
    }

    TriMesh out;
    out.V = std::move(V);
    out.F = std::move(F);
    return out;
}

}  // namespace chladni::mesh
