// ---------------------------------------------------------------------------
// chladni_figures — render a 6x6 contact sheet of Chladni nodal patterns.
//
// Solves the lowest free-edge bending modes of a circular (hex-lattice) disk
// with the Loop-subdivision shell solver, extracts the out-of-plane modal
// displacement, and draws each mode's nodal set (its zero level set — where
// sand would collect on a real Chladni plate) as glowing curves on a dark
// field. The 36 tiles are composed into one image, written as a binary PPM
// (convert to PNG with `sips -s format png out.ppm --out out.png` on macOS).
//
// Usage:
//   chladni_figures [out.ppm] [n_layers] [tile_px]
// Defaults: chladni.ppm, 30 hex layers, 340 px tiles.
// ---------------------------------------------------------------------------
#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/lme.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr int kGrid    = 6;            // 6x6 contact sheet
constexpr int kNModes  = kGrid * kGrid; // 36 figures

struct Rgb {
    float r{}, g{}, b{};
};

inline Rgb operator+(Rgb a, Rgb b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
inline Rgb operator*(Rgb a, float s) { return {a.r * s, a.g * s, a.b * s}; }

// Map a normalized signed amplitude `s` in [-1,1] plus a glow strength to the
// final tile colour: a faint sign-tinted dark body with bright cyan nodal
// curves on top.
Rgb shade(float s, float glow, float core)
{
    const float a = std::min(1.0F, std::abs(s));
    const Rgb base{6.0F, 8.0F, 15.0F};
    const Rgb cool{8.0F, 26.0F, 52.0F};   // negative lobes
    const Rgb warm{46.0F, 18.0F, 30.0F};  // positive lobes
    Rgb body = base + (s < 0.0F ? cool : warm) * a;

    const Rgb bloom{60.0F, 170.0F, 255.0F};
    const Rgb white{210.0F, 240.0F, 255.0F};
    Rgb out = body + bloom * (glow * 1.4F) + white * core;

    out.r = std::min(255.0F, out.r);
    out.g = std::min(255.0F, out.g);
    out.b = std::min(255.0F, out.b);
    return out;
}

// Separable box blur (a couple of iterations) used to turn the 1-pixel nodal
// mask into a soft bloom.
void box_blur(std::vector<float>& f, int w, int h, int radius, int iters)
{
    std::vector<float> tmp(f.size());
    const float norm = 1.0F / static_cast<float>(2 * radius + 1);
    for (int it = 0; it < iters; ++it) {
        // horizontal
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc = 0.0F;
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = std::clamp(x + dx, 0, w - 1);
                    acc += f[static_cast<std::size_t>(y * w + xx)];
                }
                tmp[static_cast<std::size_t>(y * w + x)] = acc * norm;
            }
        }
        // vertical
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc = 0.0F;
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int yy = std::clamp(y + dy, 0, h - 1);
                    acc += tmp[static_cast<std::size_t>(yy * w + x)];
                }
                f[static_cast<std::size_t>(y * w + x)] = acc * norm;
            }
        }
    }
}

// Rasterize one triangle into the per-tile field, interpolating the per-vertex
// scalar barycentrically. Pixels covered by any triangle are marked.
void raster_triangle(std::vector<float>& field, std::vector<char>& covered,
                     int w, int h,
                     float x0, float y0, float s0,
                     float x1, float y1, float s1,
                     float x2, float y2, float s2)
{
    const float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (std::abs(area) < 1e-9F) return;
    const float inv = 1.0F / area;

    int minx = static_cast<int>(std::floor(std::min({x0, x1, x2})));
    int maxx = static_cast<int>(std::ceil(std::max({x0, x1, x2})));
    int miny = static_cast<int>(std::floor(std::min({y0, y1, y2})));
    int maxy = static_cast<int>(std::ceil(std::max({y0, y1, y2})));
    minx = std::clamp(minx, 0, w - 1);
    maxx = std::clamp(maxx, 0, w - 1);
    miny = std::clamp(miny, 0, h - 1);
    maxy = std::clamp(maxy, 0, h - 1);

    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            const float px = static_cast<float>(x) + 0.5F;
            const float py = static_cast<float>(y) + 0.5F;
            float w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv;
            float w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv;
            float w2 = 1.0F - w0 - w1;
            const float eps = -1e-4F;
            if (w0 < eps || w1 < eps || w2 < eps) continue;
            const std::size_t idx = static_cast<std::size_t>(y * w + x);
            field[idx]   = w0 * s0 + w1 * s1 + w2 * s2;
            covered[idx] = 1;
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string out_path = (argc > 1) ? argv[1] : "chladni.ppm";
    const int n_layers = (argc > 2) ? std::atoi(argv[2]) : 30;
    const int tile     = (argc > 3) ? std::atoi(argv[3]) : 340;

    // Validate the numeric args: std::atoi returns 0 on garbage, and a
    // negative/zero count would mean an empty mesh or a huge image buffer.
    if (n_layers < 2 || n_layers > 200) {
        std::cerr << "n_layers must be in [2, 200] (got " << n_layers << ")\n";
        return 1;
    }
    if (tile < 32 || tile > 2048) {
        std::cerr << "tile_px must be in [32, 2048] (got " << tile << ")\n";
        return 1;
    }

    // --- geometry + physics -------------------------------------------------
    // Free-edge steel disk. The hex lattice is all-regular in the interior
    // (valence 6) with a valence-3/4 rim, so Loop subdivision accepts it and
    // its compact stencil keeps the eigensolve fast and light. (The meshfree
    // LME basis is the disk's most accurate formulation, but its wide k-ring
    // stencil makes the stiffness factorization too memory-heavy to render at
    // this resolution; the low-mode nodal *patterns* are identical either way.)
    const double radius    = 0.10;    // 10 cm
    const double thickness = 1.0e-3;  // 1 mm
    const chladni::IsotropicMaterial steel{
        .youngs_modulus = 200.0e9, .poisson_ratio = 0.33, .density = 7850.0};

    std::cerr << "Generating hex disk (" << n_layers << " layers)...\n";
    const chladni::mesh::TriMesh disk =
        chladni::mesh::generate_disk_hex(radius, n_layers);
    std::cerr << "  V=" << disk.V.rows() << " F=" << disk.F.rows() << "\n";

    // Solve extra modes so we can drop the in-plane / membrane modes (which
    // have ~zero out-of-plane displacement and would render as noise) and keep
    // the first 36 bending modes — the actual Chladni figures.
    const std::size_t n_solve = kNModes + 24;
    std::cerr << "Solving " << n_solve << " modes (Loop subdivision)...\n";
    const chladni::shell::ShellModes modes = chladni::shell::compute_shell_modes_loop(
        disk.V, disk.F, steel, thickness, n_solve);
    const Eigen::Index n_have = modes.omegas.size();

    // Keep out-of-plane-dominant (bending) modes: z-energy fraction high.
    std::vector<Eigen::Index> keep;
    for (Eigen::Index k = 0; k < n_have && static_cast<int>(keep.size()) < kNModes; ++k) {
        double oop = 0.0, tot = 0.0;
        for (Eigen::Index i = 0; i < disk.V.rows(); ++i) {
            const double x = modes.shapes(3 * i + 0, k);
            const double y = modes.shapes(3 * i + 1, k);
            const double z = modes.shapes(3 * i + 2, k);
            oop += z * z;
            tot += x * x + y * y + z * z;
        }
        if (tot > 0.0 && oop / tot > 0.6) keep.push_back(k);
    }
    std::cerr << "  kept " << keep.size() << " bending modes of " << n_have << "\n";

    // --- image layout -------------------------------------------------------
    const int gutter = tile / 24;
    const int margin = tile / 12;
    const int inset  = tile / 28;          // disk inset inside each tile
    const int draw   = tile - 2 * inset;   // disk render region (px)
    const int W = 2 * margin + kGrid * tile + (kGrid - 1) * gutter;
    const int H = W;
    const std::size_t npix = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
    std::vector<std::uint8_t> img(npix * 3, 0);
    const Rgb page{4.0F, 5.0F, 9.0F};
    for (std::size_t p = 0; p < npix; ++p) {
        img[p * 3 + 0] = static_cast<std::uint8_t>(page.r);
        img[p * 3 + 1] = static_cast<std::uint8_t>(page.g);
        img[p * 3 + 2] = static_cast<std::uint8_t>(page.b);
    }

    // Plate -> local draw-region pixel coords (y flipped for image space).
    const float drawf = static_cast<float>(draw);
    auto to_px = [&](double x, double y, float& fx, float& fy) {
        fx = static_cast<float>((x + radius) / (2.0 * radius)) * drawf;
        fy = (1.0F - static_cast<float>((y + radius) / (2.0 * radius))) * drawf;
    };

    const std::size_t ntile = static_cast<std::size_t>(draw) * static_cast<std::size_t>(draw);
    std::vector<float> field(ntile);
    std::vector<char>  covered(ntile);
    std::vector<float> nodal(ntile);

    const int n_tiles = static_cast<int>(keep.size());
    for (int slot = 0; slot < n_tiles; ++slot) {
        const Eigen::Index k = keep[static_cast<std::size_t>(slot)];
        // out-of-plane (z) displacement per vertex; the plate lies in z=0.
        double smax = 0.0;
        for (Eigen::Index i = 0; i < disk.V.rows(); ++i)
            smax = std::max(smax, std::abs(modes.shapes(3 * i + 2, k)));
        if (smax <= 0.0) smax = 1.0;
        const double inv_smax = 1.0 / smax;

        std::fill(field.begin(), field.end(), 0.0F);
        std::fill(covered.begin(), covered.end(), char{0});

        for (Eigen::Index t = 0; t < disk.F.rows(); ++t) {
            const int a = disk.F(t, 0), b = disk.F(t, 1), c = disk.F(t, 2);
            float ax, ay, bx, by, cx, cy;
            to_px(disk.V(a, 0), disk.V(a, 1), ax, ay);
            to_px(disk.V(b, 0), disk.V(b, 1), bx, by);
            to_px(disk.V(c, 0), disk.V(c, 1), cx, cy);
            raster_triangle(field, covered, draw, draw, ax, ay,
                            static_cast<float>(modes.shapes(3 * a + 2, k) * inv_smax),
                            bx, by,
                            static_cast<float>(modes.shapes(3 * b + 2, k) * inv_smax),
                            cx, cy,
                            static_cast<float>(modes.shapes(3 * c + 2, k) * inv_smax));
        }

        // Nodal set = zero crossings between covered neighbours (crisp lines,
        // independent of how large the local amplitude is).
        std::fill(nodal.begin(), nodal.end(), 0.0F);
        for (int y = 0; y < draw; ++y) {
            for (int x = 0; x < draw; ++x) {
                const std::size_t idx = static_cast<std::size_t>(y * draw + x);
                if (!covered[idx]) continue;
                const float s = field[idx];
                bool crossing = false;
                const int dxs[4] = {1, -1, 0, 0};
                const int dys[4] = {0, 0, 1, -1};
                for (int n = 0; n < 4 && !crossing; ++n) {
                    const int xx = x + dxs[n], yy = y + dys[n];
                    if (xx < 0 || xx >= draw || yy < 0 || yy >= draw) continue;
                    const std::size_t j = static_cast<std::size_t>(yy * draw + xx);
                    if (covered[j] && field[j] * s < 0.0F) crossing = true;
                }
                if (crossing) nodal[idx] = 1.0F;
            }
        }

        // Bloom: a blurred copy of the crisp nodal mask.
        std::vector<float> bloom = nodal;
        box_blur(bloom, draw, draw, std::max(1, draw / 200), 2);
        float bmax = 0.0F;
        for (float v : bloom) bmax = std::max(bmax, v);
        if (bmax <= 0.0F) bmax = 1.0F;

        // Compose this tile into the full image.
        const int gx = slot % kGrid;
        const int gy = slot / kGrid;
        const int ox = margin + gx * (tile + gutter) + inset;
        const int oy = margin + gy * (tile + gutter) + inset;
        for (int y = 0; y < draw; ++y) {
            for (int x = 0; x < draw; ++x) {
                const std::size_t idx = static_cast<std::size_t>(y * draw + x);
                if (!covered[idx]) continue;  // outside the disk -> page bg
                const float gl = bloom[idx] / bmax;
                const Rgb col = shade(field[idx], gl, nodal[idx] * 0.9F);
                const std::size_t px =
                    static_cast<std::size_t>((oy + y) * W + (ox + x)) * 3;
                img[px + 0] = static_cast<std::uint8_t>(col.r);
                img[px + 1] = static_cast<std::uint8_t>(col.g);
                img[px + 2] = static_cast<std::uint8_t>(col.b);
            }
        }
        std::cerr << "  tile " << (slot + 1) << "/" << n_tiles << " ("
                  << modes.omegas(k) / (2.0 * std::numbers::pi_v<double>)
                  << " Hz)\n";
    }

    // --- write binary PPM ---------------------------------------------------
    std::ofstream f(out_path, std::ios::binary);
    if (!f) {
        std::cerr << "error: cannot open " << out_path << " for writing\n";
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()),
            static_cast<std::streamsize>(img.size()));
    std::cerr << "Wrote " << out_path << " (" << W << "x" << H << ")\n";
    return 0;
}
