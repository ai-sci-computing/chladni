/**
 * @file test_lme_shepard.cpp
 * @brief Unit tests for the Shepard partition-of-unity helper used by
 *        the curved-shell LME assembler.
 *
 * Properties verified, from
 * @cite millan_rosolen_arroyo_2011_thin_shell_maxent §2 eqs. (1)–(3):
 *
 *  1. **PoU identity** — @f$ \sum_A w_A^Q(x) = 1 @f$ at any point where
 *     at least one patch is in range. Independent of the value of
 *     @f$ \beta_A @f$.
 *  2. **Truncation** — patches whose normalised weight falls below the
 *     caller-supplied @c tol are dropped from the returned support. The
 *     surviving weights still sum to 1 (renormalisation after drop).
 *  3. **Concentration at large @f$ \beta @f$** — as the locality
 *     parameter grows the Shepard PU concentrates on the patch with
 *     the smallest @f$ |x - Q_A|^2 @f$. Verifies the Gaussian-weight
 *     ranking.
 *  4. **Input validation** — empty patch set, mismatched @c beta, and
 *     non-positive @c tol must throw.
 */

#include <chladni/shell/lme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>

#include <cmath>
#include <numeric>
#include <vector>

using chladni::shell::lme::shepard_partition;
using chladni::shell::lme::ShepardWeights;
using Catch::Matchers::WithinAbs;

namespace {

/// 4 patch anchors at the corners of a unit square in the z=0 plane.
Eigen::MatrixXd four_corner_patches()
{
    Eigen::MatrixXd Q(4, 3);
    Q << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         1.0, 1.0, 0.0;
    return Q;
}

}  // namespace

TEST_CASE("lme::shepard_partition: PoU identity at interior query",
          "[shell][lme][shepard]")
{
    const auto Q    = four_corner_patches();
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(Q.rows(), 4.0);
    Eigen::Vector3d x(0.4, 0.6, 0.0);  // strictly inside the unit square

    const ShepardWeights w = shepard_partition(Q, beta, x, /*tol=*/1e-12);

    REQUIRE(w.indices.size() == w.values.size());
    REQUIRE(!w.values.empty());

    const double sum =
        std::accumulate(w.values.begin(), w.values.end(), 0.0);
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-12));

    for (double v : w.values) REQUIRE(v >= 0.0);
}

TEST_CASE("lme::shepard_partition: truncation drops below-tol patches",
          "[shell][lme][shepard]")
{
    const auto Q    = four_corner_patches();
    // Large beta + a point much closer to one corner means the other
    // three weights are exponentially small. With tol=1e-2 the support
    // should collapse to a single patch.
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(Q.rows(), 200.0);
    Eigen::Vector3d x(0.02, 0.0, 0.0);  // right next to corner 0

    const ShepardWeights w = shepard_partition(Q, beta, x, /*tol=*/1e-2);

    REQUIRE(w.indices.size() == 1);
    REQUIRE(w.indices.front() == 0);
    REQUIRE_THAT(w.values.front(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("lme::shepard_partition: concentration on nearest patch as β→∞",
          "[shell][lme][shepard]")
{
    const auto Q    = four_corner_patches();
    // At (0.4, 0.4, 0) the nearest patch is corner 0 (distance √0.32 ≈
    // 0.566) — the others are at 0.721, 0.721, 1.0. With very large β
    // the weight ratios w_other / w_0 decay like exp(-β · Δd²) and the
    // nearest patch should dominate. We don't push to exact 1 because
    // the second-nearest patches are only marginally further.
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(Q.rows(), 50.0);
    Eigen::Vector3d x(0.4, 0.4, 0.0);

    const ShepardWeights w = shepard_partition(Q, beta, x, /*tol=*/1e-14);

    // Locate the entry corresponding to patch 0 — it should be largest.
    double max_v   = -1.0;
    int    max_idx = -1;
    for (std::size_t k = 0; k < w.values.size(); ++k) {
        if (w.values[k] > max_v) {
            max_v   = w.values[k];
            max_idx = w.indices[k];
        }
    }
    REQUIRE(max_idx == 0);
    REQUIRE(max_v > 0.95);
}

TEST_CASE("lme::shepard_partition: empty patch set throws",
          "[shell][lme][shepard]")
{
    Eigen::MatrixXd Q(0, 3);
    Eigen::VectorXd beta(0);
    Eigen::Vector3d x(0.0, 0.0, 0.0);
    REQUIRE_THROWS_AS(
        shepard_partition(Q, beta, x, /*tol=*/1e-10),
        std::invalid_argument);
}

TEST_CASE("lme::shepard_partition: beta size mismatch throws",
          "[shell][lme][shepard]")
{
    const auto Q   = four_corner_patches();
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(Q.rows() - 1, 1.0);
    Eigen::Vector3d x(0.5, 0.5, 0.0);
    REQUIRE_THROWS_AS(
        shepard_partition(Q, beta, x, /*tol=*/1e-10),
        std::invalid_argument);
}

TEST_CASE("lme::shepard_partition: non-positive tol throws",
          "[shell][lme][shepard]")
{
    const auto Q    = four_corner_patches();
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(Q.rows(), 1.0);
    Eigen::Vector3d x(0.5, 0.5, 0.0);
    REQUIRE_THROWS_AS(
        shepard_partition(Q, beta, x, /*tol=*/0.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        shepard_partition(Q, beta, x, /*tol=*/-1e-12),
        std::invalid_argument);
}
