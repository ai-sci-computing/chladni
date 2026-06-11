/**
 * @file test_sme_brezel_roughness.cpp
 * @brief Diagnose the jagged SME nodal lines reported on the genus-2
 *        pretzel (models/brezel2.obj) in the GUI (2026-06-08).
 *
 * The user rendered SME mode 27 (~4725 Hz) on brezel2 (2622 V, closed)
 * and saw nodal lines decorated with jagged staircases + scattered tiny
 * "diamond/triangle" sign-islands — the signature of high-frequency
 * noise in the modal field, NOT a clean Chladni pattern.
 *
 * This test pins down WHETHER it is SME-specific and WHY, by
 * construction rather than eyeballing:
 *
 *  1. MESH QUALITY — valence histogram + worst triangle angle of the
 *     loaded mesh (an irregular load, outside every analytic fixture).
 *  2. SCAFFOLDING — does SME's B1 drop-net / B3 escalation FIRE on this
 *     real mesh? (last_drop_stats after assemble_K). If the invented
 *     feasibility machinery is active here, the field carries it.
 *  3. ROUGHNESS — for Loop / LME-1st / SME, count nodal-domain
 *     structure of each mode's NORMAL displacement: total sign-domains
 *     and the number of TINY ones (≤2 verts) — the tiny islands ARE
 *     the diamonds in the screenshot. A clean mode-k has O(k) domains
 *     and ~0 tiny ones; a noisy field inflates the tiny count. The
 *     discriminator: is SME's tiny-island count much higher than
 *     Loop's / LME's at matched mode index?
 *
 * Tagged [.diag][.slow] — run on demand:
 *   ./build/tests/chladni_tests "[sme_brezel]"
 */

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/lme.hpp>
#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <numbers>
#include <vector>

namespace {

namespace fs = std::filesystem;

fs::path data_dir()
{
    if (const char* env = std::getenv("CHLADNI_DATA_DIR"))
        return fs::path{env};
    return fs::path{CHLADNI_DATA_DIR};
}

/// Area-weighted vertex normals for the rest mesh.
Eigen::MatrixXd vertex_normals(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F)
{
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(V.rows(), 3);
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
        const Eigen::Vector3d a = V.row(F(f, 0));
        const Eigen::Vector3d b = V.row(F(f, 1));
        const Eigen::Vector3d c = V.row(F(f, 2));
        const Eigen::Vector3d e1 = b - a;
        const Eigen::Vector3d e2 = c - a;
        const Eigen::Vector3d n = e1.cross(e2);  // area-weighted
        for (int k = 0; k < 3; ++k) N.row(F(f, k)) += n.transpose();
    }
    for (Eigen::Index v = 0; v < V.rows(); ++v) {
        const double nn = N.row(v).norm();
        if (nn > 0) N.row(v) /= nn;
    }
    return N;
}

/// Vertex adjacency from faces.
std::vector<std::vector<int>> adjacency(const Eigen::MatrixXi& F,
                                        Eigen::Index n_v)
{
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n_v));
    auto add = [&](int a, int b) {
        auto& va = adj[static_cast<std::size_t>(a)];
        if (std::find(va.begin(), va.end(), b) == va.end()) va.push_back(b);
    };
    for (Eigen::Index f = 0; f < F.rows(); ++f)
        for (int e = 0; e < 3; ++e) {
            add(F(f, e), F(f, (e + 1) % 3));
            add(F(f, (e + 1) % 3), F(f, e));
        }
    return adj;
}

/// Count connected same-sign nodal domains of a per-vertex scalar, and
/// how many are "tiny" (≤ max_tiny vertices — the spurious islands).
struct DomainStats {
    int total = 0;
    int tiny  = 0;   ///< domains of ≤ max_tiny vertices
    int largest = 0;
};

DomainStats nodal_domains(const Eigen::VectorXd& w,
                          const std::vector<std::vector<int>>& adj,
                          int max_tiny = 2)
{
    const int n = static_cast<int>(w.size());
    std::vector<int> comp(static_cast<std::size_t>(n), -1);
    DomainStats s;
    for (int seed = 0; seed < n; ++seed) {
        if (comp[static_cast<std::size_t>(seed)] != -1) continue;
        const bool sign = w(seed) >= 0.0;
        // BFS over same-sign neighbours.
        std::vector<int> stack{seed};
        comp[static_cast<std::size_t>(seed)] = s.total;
        int size = 0;
        while (!stack.empty()) {
            const int u = stack.back(); stack.pop_back();
            ++size;
            for (int v : adj[static_cast<std::size_t>(u)]) {
                if (comp[static_cast<std::size_t>(v)] != -1) continue;
                if ((w(v) >= 0.0) != sign) continue;
                comp[static_cast<std::size_t>(v)] = s.total;
                stack.push_back(v);
            }
        }
        ++s.total;
        if (size <= max_tiny) ++s.tiny;
        s.largest = std::max(s.largest, size);
    }
    return s;
}

/// Per-vertex normal displacement of mode k (column k of shapes).
Eigen::VectorXd normal_disp(const chladni::shell::ShellModes& m,
                            const Eigen::MatrixXd& N, int k)
{
    const Eigen::Index n_v = N.rows();
    Eigen::VectorXd w(n_v);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        const Eigen::Vector3d d = m.shapes.block(3 * v, k, 3, 1);
        w(v) = d.dot(N.row(v).transpose());
    }
    return w;
}

}  // namespace

TEST_CASE("SME brezel roughness — is the jagged nodal field SME-specific?",
          "[.diag][.slow][shell][sme][sme_brezel]")
{
    const auto obj_path = data_dir() / "brezel2.obj";
    if (!std::filesystem::exists(obj_path))
        SKIP("models/brezel2.obj is not bundled in this distribution");
    const auto mesh = chladni::mesh::load_obj(obj_path);
    const Eigen::Index n_v = mesh.V.rows();
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9, .poisson_ratio = 1.0 / 3.0,
        .density = 7850.0};
    constexpr double h = 1.0e-3;
    const auto sm = chladni::shell::shell_material_from_isotropic(steel, h);
    constexpr int n_modes = 30;

    // ---- 1. Mesh quality. ------------------------------------------
    const auto adj = adjacency(mesh.F, n_v);
    std::map<int, int> vhist;
    for (Eigen::Index v = 0; v < n_v; ++v)
        ++vhist[static_cast<int>(adj[static_cast<std::size_t>(v)].size())];
    double min_angle = 180.0;
    for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
        const Eigen::Vector3d p[3] = {mesh.V.row(mesh.F(f, 0)),
                                      mesh.V.row(mesh.F(f, 1)),
                                      mesh.V.row(mesh.F(f, 2))};
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector3d a = (p[(i + 1) % 3] - p[i]).normalized();
            const Eigen::Vector3d b = (p[(i + 2) % 3] - p[i]).normalized();
            const double ang = std::acos(std::clamp(a.dot(b), -1.0, 1.0))
                               * 180.0 / std::numbers::pi;
            min_angle = std::min(min_angle, ang);
        }
    }
    std::fprintf(stderr,
        "\n[sme_brezel] brezel2.obj: %lld V, %lld F, worst tri angle "
        "%.1f deg\n[sme_brezel] valence hist:",
        static_cast<long long>(n_v),
        static_cast<long long>(mesh.F.rows()), min_angle);
    for (const auto& [val, c] : vhist)
        std::fprintf(stderr, " v%d=%d", val, c);
    std::fprintf(stderr, "\n");

    // ---- 2. Does SME's scaffolding fire on this real mesh? ----------
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.use_ghost_nodes      = false;  // closed mesh
        chladni::shell::LMEAssembler a{p};
        (void)a.assemble_K(mesh.V, mesh.F, sm);
        const auto& ds = a.last_drop_stats();
        std::fprintf(stderr,
            "[sme_brezel] SME assemble_K drop stats: ownership n=%ld "
            "(w=%.3g)  newton n=%ld (w=%.3g)  escal_ok=%ld  "
            "w_gauss_max=%.3g\n",
            ds.n_own, ds.w_own, ds.n_newton, ds.w_newton,
            ds.n_escal_ok, ds.w_gauss_max);
    }

    // ---- 3. Modal roughness + checkerboard spurious-mode test. -----
    const Eigen::MatrixXd N = vertex_normals(mesh.V, mesh.F);

    // The consistent-diagonal quad triangulation 2-colours the vertices:
    // diagonal-carrying = valence 8/10, non-carrying = valence 4. The
    // alternating ±normal displacement on those colours IS the
    // checkerboard / hourglass pattern. Its Rayleigh quotient
    // ω²=φᵀKφ/φᵀMφ measures how STIFF each method is to that pattern —
    // if meshfree under-penalises it (low ω vs Loop), it leaks into the
    // low modes and shows up as ripple near nodal lines (the diamonds).
    Eigen::VectorXd checker(3 * n_v);
    for (Eigen::Index v = 0; v < n_v; ++v) {
        const double s =
            adj[static_cast<std::size_t>(v)].size() >= 8 ? 1.0 : -1.0;
        checker.segment<3>(3 * v) = s * N.row(v).transpose();
    }
    const double rho_h = steel.density * h;

    auto analyse = [&](const char* tag,
                       const chladni::shell::ShellAssembler& A) {
        const auto m = chladni::shell::compute_shell_modes(
            mesh.V, mesh.F, steel, sm, h, n_modes, A);
        const double f0 = m.omegas(0) / (2.0 * std::numbers::pi);

        // Roughness: tiny-island counts at LOW (0–9) and BAND (20–29)
        // modes. Loop's tiny count should GROW with mode (nodal-line
        // length); genuine ripple shows up even at LOW modes.
        //
        // Vector Dirichlet roughness R = Σ_edges‖d_i−d_j‖² /
        // Σ_i deg_i‖d_i‖² — NORMAL-INDEPENDENT (raw 3-vector field, no
        // projection), so it isolates BASIS smoothness from any
        // vertex-normal noise. A smooth mode has small R; ripple
        // inflates it. Printed for the FUNDAMENTAL (mode 0) and a few
        // low modes: aliasing cannot explain a rough fundamental, so a
        // high R at mode 0 is the basis itself.
        auto dirichlet = [&](int k) {
            double num = 0.0, den = 0.0;
            for (Eigen::Index v = 0; v < n_v; ++v) {
                const Eigen::Vector3d dv = m.shapes.block(3 * v, k, 3, 1);
                den += static_cast<double>(
                           adj[static_cast<std::size_t>(v)].size())
                       * dv.squaredNorm();
                for (int u : adj[static_cast<std::size_t>(v)])
                    if (u > v) {
                        const Eigen::Vector3d du =
                            m.shapes.block(3 * u, k, 3, 1);
                        num += (dv - du).squaredNorm();
                    }
            }
            return den > 0 ? num / den : 0.0;
        };
        int lo_tiny_sum = 0, band_tiny_sum = 0, worst = 0, worst_k = -1;
        for (int k = 0; k < n_modes; ++k) {
            const DomainStats d = nodal_domains(normal_disp(m, N, k), adj);
            if (d.tiny > worst) { worst = d.tiny; worst_k = k; }
            if (k < 10)            lo_tiny_sum   += d.tiny;
            if (k >= 20 && k < 30) band_tiny_sum += d.tiny;
        }
        std::fprintf(stderr,
            "[sme_brezel] %-8s  Dirichlet roughness R (normal-free): "
            "mode0=%.3f mode1=%.3f mode2=%.3f | mode0 tiny-islands=%d\n",
            tag, dirichlet(0), dirichlet(1), dirichlet(2),
            nodal_domains(normal_disp(m, N, 0), adj).tiny);

        // Checkerboard Rayleigh quotient on this method's own K, M.
        const Eigen::SparseMatrix<double> K = A.assemble_K(mesh.V, mesh.F, sm);
        const Eigen::SparseMatrix<double> M =
            A.assemble_M(mesh.V, mesh.F, rho_h);
        double f_check = std::nan("");
        if (K.rows() == 3 * n_v && M.rows() == 3 * n_v) {
            const double num = checker.dot(K * checker);
            const double den = checker.dot(M * checker);
            if (den > 0 && num >= 0)
                f_check = std::sqrt(num / den) / (2.0 * std::numbers::pi);
        }
        std::fprintf(stderr,
            "[sme_brezel] %-8s  mode0 f=%.1f Hz | tiny-islands: "
            "low(0-9) sum=%4d  band(20-29) sum=%4d  worst=%3d@mode%d | "
            "checkerboard f=%.1f Hz  (ratio to mode0 = %.1f×)\n",
            tag, f0, lo_tiny_sum, band_tiny_sum, worst, worst_k,
            f_check, f_check / f0);
    };

    {
        chladni::shell::LoopAssembler loop;
        analyse("Loop", loop);
    }
    {
        chladni::shell::LMEAssembler lme{
            chladni::shell::LMEAssembler::Params{}};
        analyse("LME-1st", lme);
    }
    {
        chladni::shell::LMEAssembler::Params p;
        p.use_second_order_sme = true;
        p.use_ghost_nodes      = false;
        chladni::shell::LMEAssembler sme{p};
        analyse("SME", sme);
    }

    SUCCEED("brezel roughness + checkerboard RQ printed");
}
