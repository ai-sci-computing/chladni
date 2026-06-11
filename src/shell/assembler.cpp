/**
 * @file assembler.cpp
 * @brief Concrete `LoopAssembler` wrapping the Cirak-Ortiz Loop FEM.
 *
 * Thin adapter: each `assemble_K` / `assemble_M` call delegates to the
 * pre-existing free-function pipeline in
 * @ref chladni::shell::loop::assemble_stiffness_loop /
 * @ref chladni::shell::loop::assemble_mass_loop, then applies the
 * post-assembly `MassLumping` policy on `M`.
 *
 * The adapter forwards @ref chladni::shell::MassLumping, the @c n_passes / @c use_stam
 * knobs, and the per-element @ref chladni::shell::QuadratureRule (@c k_quad / @c m_quad)
 * straight through to the Loop kernels, which evaluate
 * @ref chladni::shell::quadrature_points for whichever rule is selected —
 * every rule is honored, none is rejected. The constructor validates
 * only @c n_passes (>= 1).
 */

#include <chladni/shell/assembler.hpp>

#include <chladni/shell/loop.hpp>

#include <stdexcept>
#include <vector>

namespace chladni::shell {

std::vector<QuadraturePoint> quadrature_points(QuadratureRule rule)
{
    switch (rule) {
    case QuadratureRule::OnePointCentroid:
        // Single sample at the centroid. Exact for degree <= 1.
        // Weight equals the area of the reference unit-area triangle
        // (1/2). Cirak-Ortiz Sec 4.6 recommends this rule for the
        // plate-bending statics problem on the 12-DOF box-spline basis;
        // it preserves convergence ORDER on K (the bending integrand's
        // degree-4 leading term is captured asymptotically) but leaves
        // a larger per-element residual at finite h than the higher-
        // order rules below.
        return { QuadraturePoint{1.0 / 3.0, 1.0 / 3.0, 0.5} };

    case QuadratureRule::ThreePointEdgeMid:
        // Three samples at the edge midpoints. Exact for degree <= 2.
        // Equal weights summing to 1/2. Classical degree-2 rule.
        return {
            QuadraturePoint{0.5, 0.0, 1.0 / 6.0},
            QuadraturePoint{0.0, 0.5, 1.0 / 6.0},
            QuadraturePoint{0.5, 0.5, 1.0 / 6.0},
        };

    case QuadratureRule::SevenPointDunavant: {
        // Dunavant 1985 degree-5 symmetric rule on the unit-area
        // triangle. Centroid + two three-point orbits. Weights below
        // are Dunavant Table II values divided by 2 so that they sum
        // to 1/2 (the area of the reference triangle), matching the
        // convention used by the other rules above. Integrates degree-5
        // polynomials exactly — comfortably above the degree-4 Loop
        // box-spline basis.
        constexpr double alpha_1 = 0.7974269853530873;
        constexpr double beta_1  = 0.1012865073234563;
        constexpr double alpha_2 = 0.0597158717897698;
        constexpr double beta_2  = 0.4701420641051151;
        constexpr double w_c     = 0.1125;
        constexpr double w_1     = 0.06296959027241353;
        constexpr double w_2     = 0.06619707639425308;
        return {
            QuadraturePoint{1.0 / 3.0, 1.0 / 3.0, w_c},
            QuadraturePoint{beta_1,    beta_1,    w_1},
            QuadraturePoint{alpha_1,   beta_1,    w_1},
            QuadraturePoint{beta_1,    alpha_1,   w_1},
            QuadraturePoint{beta_2,    beta_2,    w_2},
            QuadraturePoint{alpha_2,   beta_2,    w_2},
            QuadraturePoint{beta_2,    alpha_2,   w_2},
        };
    }

    case QuadratureRule::TwelvePointDunavant: {
        // Dunavant 1985 degree-6 symmetric rule (12 points) on the
        // unit-area triangle — the rule the max-ent shell paper uses
        // (Millán 2011 §4.1.1, "12 points (order 6) per triangle").
        // Three orbits; barycentric (1-v-w, v, w). Table II weights
        // divided by 2 so they sum to 1/2 (reference-triangle area).
        constexpr double a   = 0.063089014491502;   // orbit (a,a,1-2a)
        constexpr double b   = 0.249286745170910;   // orbit (b,b,1-2b)
        constexpr double c   = 0.310352451033785;   // orbit (c,d,e) ×6
        constexpr double d   = 0.053145049844816;
        constexpr double e   = 1.0 - c - d;
        constexpr double w_a  = 0.050844906370207 / 2.0;
        constexpr double w_b  = 0.116786275726379 / 2.0;
        constexpr double w_cd = 0.082851075618374 / 2.0;
        return {
            QuadraturePoint{a,       a,       w_a},
            QuadraturePoint{1.0-2*a, a,       w_a},
            QuadraturePoint{a,       1.0-2*a, w_a},
            QuadraturePoint{b,       b,       w_b},
            QuadraturePoint{1.0-2*b, b,       w_b},
            QuadraturePoint{b,       1.0-2*b, w_b},
            QuadraturePoint{d,       e,       w_cd},
            QuadraturePoint{e,       d,       w_cd},
            QuadraturePoint{c,       e,       w_cd},
            QuadraturePoint{e,       c,       w_cd},
            QuadraturePoint{c,       d,       w_cd},
            QuadraturePoint{d,       c,       w_cd},
        };
    }
    }
    throw std::logic_error("quadrature_points: unknown QuadratureRule");
}

namespace {

Eigen::SparseMatrix<double> row_sum_lump(
    const Eigen::SparseMatrix<double>& M_full)
{
    const Eigen::VectorXd row_sums =
        M_full * Eigen::VectorXd::Ones(M_full.cols());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(M_full.rows()));
    for (Eigen::Index i = 0; i < M_full.rows(); ++i) {
        const double m_ii = row_sums(i);
        if (m_ii > 0.0) {
            trips.emplace_back(i, i, m_ii);
        }
    }
    Eigen::SparseMatrix<double> M_lumped(M_full.rows(), M_full.cols());
    M_lumped.setFromTriplets(trips.begin(), trips.end());
    M_lumped.makeCompressed();
    return M_lumped;
}

}  // namespace

LoopAssembler::LoopAssembler(Params p) : params_(p)
{
    if (params_.n_passes < 1) {
        throw std::invalid_argument(
            "LoopAssembler: Params::n_passes must be >= 1");
    }
}

Eigen::SparseMatrix<double> LoopAssembler::assemble_K(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const ShellMaterial&   material) const
{
    return chladni::shell::loop::assemble_stiffness_loop(
        V, F, material, params_.n_passes, params_.use_stam, params_.k_quad);
}

Eigen::SparseMatrix<double> LoopAssembler::assemble_M(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double                 surface_density) const
{
    auto M_full = chladni::shell::loop::assemble_mass_loop(
        V, F, surface_density, params_.n_passes, params_.use_stam,
        params_.m_quad);

    switch (params_.m_lump) {
    case MassLumping::None:
        return M_full;
    case MassLumping::RowSum:
        return row_sum_lump(M_full);
    }
    // Unreachable; switch is exhaustive over the enum.
    throw std::logic_error("LoopAssembler::assemble_M: unknown MassLumping");
}

std::string LoopAssembler::label() const
{
    return "Loop (Cirak-Ortiz)";
}

}  // namespace chladni::shell
