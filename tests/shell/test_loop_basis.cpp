/**
 * @file test_loop_basis.cpp
 * @brief Unit tests for the regular Loop box-spline basis (Cirak-Ortiz Eq. 75).
 *
 * Closed-form quartic polynomial basis on the unit triangle. Tests are
 * pure mathematical sanity checks — no mesh data, no I/O.
 *
 * Coverage:
 *
 * 1. **Partition of unity** at a battery of @f$(v, w)@f$ points
 *    (centroid, edge midpoints, corners, asymmetric interior points).
 *    @f$ \sum_{I=1}^{12} N_I = 1 @f$ exactly within round-off because
 *    the basis is an affinely-invariant linear combination of degree-4
 *    monomials with rational coefficients over 12.
 *
 * 2. **Specific values at the centroid** @f$(v, w) = (1/3, 1/3)@f$ by
 *    3-fold symmetry of the hexagonal stencil:
 *      - @f$N_4 = N_7 = N_8 = 23/81@f$ (the 3 "triangle-corner" basis
 *        functions, slots 3, 6, 7 in 0-indexed),
 *      - @f$N_3 = N_5 = N_{11} = 14/324@f$ (the 3 "edge-opposite"
 *        basis functions, slots 2, 4, 10),
 *      - @f$N_1 = N_2 = N_6 = N_9 = N_{10} = N_{12} = 1/324@f$ (the 6
 *        "outer-hexagon" basis functions, slots 0, 1, 5, 8, 9, 11).
 *    Total @f$ 3 \cdot 23/81 + 3 \cdot 14/324 + 6 \cdot 1/324 = 1 @f$.
 *
 * 3. **Symmetry under @f$ v \leftrightarrow w @f$** at points where the
 *    swap is a fixed point of the relevant pair of basis functions
 *    (Cirak-Ortiz Fig. 9 has the @f$v=w@f$ axis as a mirror; specific
 *    pair maps:
 *    @f$N_1 \leftrightarrow N_2@f$, @f$N_3 \leftrightarrow N_5@f$,
 *    @f$N_6 \leftrightarrow N_9@f$, @f$N_{10} \leftrightarrow N_{12}@f$,
 *    @f$N_7 \leftrightarrow N_8@f$, with @f$N_4@f$ and @f$N_{11}@f$
 *    self-symmetric).
 *
 * 4. **Non-negativity**: every basis value is @f$\geq 0@f$ on the unit
 *    triangle. Box splines are nonneg by construction.
 */

#include <chladni/shell/loop.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

#include <vector>

namespace {

/// (v, w) sample point inside the closed unit triangle.
struct Sample {
    double v;
    double w;
};

const std::vector<Sample> interior_samples = {
    {1.0 / 3.0, 1.0 / 3.0},   // centroid
    {0.5,        0.25      },  // generic interior
    {0.1,        0.6       },  // generic interior
    {0.7,        0.2       },  // generic interior
    {0.0,        0.5       },  // edge midpoint (u-w edge, v=0)
    {0.5,        0.0       },  // edge midpoint (u-v edge, w=0)
    {0.5,        0.5       },  // edge midpoint (v-w edge, u=0)
    {0.0,        0.0       },  // u=1 corner
    {1.0,        0.0       },  // v=1 corner
    {0.0,        1.0       },  // w=1 corner
};

}  // namespace

TEST_CASE("regular_basis: partition of unity on a battery of points",
          "[shell][loop][basis][partition_of_unity]")
{
    for (const auto& s : interior_samples) {
        CAPTURE(s.v, s.w);
        const auto N = chladni::shell::loop::regular_basis(s.v, s.w);
        REQUIRE(N.sum() == Catch::Approx(1.0).margin(1e-14));
    }
}

TEST_CASE("regular_basis: N_11 is self-symmetric under v <-> w",
          "[shell][loop][basis][n11_self_symmetry]")
{
    // N_11 corresponds to the vertex opposite to vertex 4 across the
    // central triangle's edge (7-8). Geometrically that vertex sits on
    // the v=w mirror axis, so the basis function should be invariant
    // under swapping v and w. This was a transcription bug in the
    // first cut of regular_basis (a 6 u^2 v w term where there should
    // have been 6 u v^2 w). Verify across several asymmetric (v, w).
    const std::vector<Sample> asymmetric = {
        {0.41, 0.18},
        {0.10, 0.65},
        {0.55, 0.05},
        {0.20, 0.20},   // here v == w, so trivially equal but still a sanity check
    };
    for (const auto& s : asymmetric) {
        CAPTURE(s.v, s.w);
        const auto Nvw = chladni::shell::loop::regular_basis(s.v, s.w);
        const auto Nwv = chladni::shell::loop::regular_basis(s.w, s.v);
        REQUIRE(Nvw(10) == Catch::Approx(Nwv(10)).margin(1e-14));
    }
}

TEST_CASE("regular_basis: centroid values match closed form (3-fold symmetry)",
          "[shell][loop][basis][centroid]")
{
    const auto N = chladni::shell::loop::regular_basis(1.0 / 3.0, 1.0 / 3.0);

    // 3 corner basis functions: N_4 = N_7 = N_8 = 23/81 (slots 3, 6, 7).
    REQUIRE(N(3) == Catch::Approx(23.0 / 81.0).margin(1e-14));
    REQUIRE(N(6) == Catch::Approx(23.0 / 81.0).margin(1e-14));
    REQUIRE(N(7) == Catch::Approx(23.0 / 81.0).margin(1e-14));

    // 3 edge-opposite basis functions: N_3 = N_5 = N_11 = 14/324 = 7/162
    // (slots 2, 4, 10).
    REQUIRE(N(2)  == Catch::Approx(7.0 / 162.0).margin(1e-14));
    REQUIRE(N(4)  == Catch::Approx(7.0 / 162.0).margin(1e-14));
    REQUIRE(N(10) == Catch::Approx(7.0 / 162.0).margin(1e-14));

    // 6 outer-hexagon basis functions: 1/324 each (slots 0, 1, 5, 8, 9, 11).
    REQUIRE(N(0)  == Catch::Approx(1.0 / 324.0).margin(1e-14));
    REQUIRE(N(1)  == Catch::Approx(1.0 / 324.0).margin(1e-14));
    REQUIRE(N(5)  == Catch::Approx(1.0 / 324.0).margin(1e-14));
    REQUIRE(N(8)  == Catch::Approx(1.0 / 324.0).margin(1e-14));
    REQUIRE(N(9)  == Catch::Approx(1.0 / 324.0).margin(1e-14));
    REQUIRE(N(11) == Catch::Approx(1.0 / 324.0).margin(1e-14));
}

TEST_CASE("regular_basis: v <-> w symmetry maps Fig. 9 vertex pairs",
          "[shell][loop][basis][symmetry]")
{
    // At an asymmetric (v, w) point, swapping v and w should permute the
    // basis functions according to the geometric mirror across the v=w
    // axis of Cirak-Ortiz Fig. 9. Pair maps:
    //   1 <-> 2     (slots 0 <-> 1)
    //   3 <-> 5     (slots 2 <-> 4)
    //   6 <-> 9     (slots 5 <-> 8)
    //   10 <-> 12   (slots 9 <-> 11)
    //   7 <-> 8     (slots 6 <-> 7)
    // and N_4, N_11 are self-symmetric.
    const double v = 0.41;
    const double w = 0.18;
    const auto Nvw = chladni::shell::loop::regular_basis(v, w);
    const auto Nwv = chladni::shell::loop::regular_basis(w, v);

    REQUIRE(Nvw(0) == Catch::Approx(Nwv(1)).margin(1e-14));
    REQUIRE(Nvw(1) == Catch::Approx(Nwv(0)).margin(1e-14));
    REQUIRE(Nvw(2) == Catch::Approx(Nwv(4)).margin(1e-14));
    REQUIRE(Nvw(4) == Catch::Approx(Nwv(2)).margin(1e-14));
    REQUIRE(Nvw(5) == Catch::Approx(Nwv(8)).margin(1e-14));
    REQUIRE(Nvw(8) == Catch::Approx(Nwv(5)).margin(1e-14));
    REQUIRE(Nvw(9) == Catch::Approx(Nwv(11)).margin(1e-14));
    REQUIRE(Nvw(11) == Catch::Approx(Nwv(9)).margin(1e-14));
    REQUIRE(Nvw(6) == Catch::Approx(Nwv(7)).margin(1e-14));
    REQUIRE(Nvw(7) == Catch::Approx(Nwv(6)).margin(1e-14));
    // Self-symmetric.
    REQUIRE(Nvw(3)  == Catch::Approx(Nwv(3)).margin(1e-14));
    REQUIRE(Nvw(10) == Catch::Approx(Nwv(10)).margin(1e-14));
}

TEST_CASE("regular_basis: non-negative on the unit triangle",
          "[shell][loop][basis][nonneg]")
{
    for (const auto& s : interior_samples) {
        const auto N = chladni::shell::loop::regular_basis(s.v, s.w);
        for (Eigen::Index i = 0; i < N.size(); ++i) {
            REQUIRE(N(i) >= -1e-15);  // allow tiny negative round-off
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient and Hessian tests.
//
// The basis values are now trusted (verified above), so the analytic
// gradient and Hessian must agree with central finite differences of
// the basis themselves, and must individually annihilate the
// partition-of-unity sum (a constant has zero derivatives).
// ---------------------------------------------------------------------------

namespace {

const std::vector<Sample> interior_only_samples = {
    {1.0 / 3.0, 1.0 / 3.0},
    {0.5,        0.25      },
    {0.25,       0.5       },
    {0.1,        0.6       },
    {0.7,        0.2       },
    {0.2,        0.2       },
    {0.4,        0.4       },
};

}  // namespace

TEST_CASE("regular_basis_grad: agrees with central FD of basis values",
          "[shell][loop][basis][grad][fd]")
{
    constexpr double h = 1.0e-5;
    for (const auto& s : interior_only_samples) {
        CAPTURE(s.v, s.w);
        const auto G = chladni::shell::loop::regular_basis_grad(s.v, s.w);
        const auto Nvp = chladni::shell::loop::regular_basis(s.v + h, s.w);
        const auto Nvm = chladni::shell::loop::regular_basis(s.v - h, s.w);
        const auto Nwp = chladni::shell::loop::regular_basis(s.v, s.w + h);
        const auto Nwm = chladni::shell::loop::regular_basis(s.v, s.w - h);
        const Eigen::Matrix<double, 12, 1> dN_dv_fd = (Nvp - Nvm) / (2.0 * h);
        const Eigen::Matrix<double, 12, 1> dN_dw_fd = (Nwp - Nwm) / (2.0 * h);
        for (Eigen::Index i = 0; i < 12; ++i) {
            CAPTURE(i);
            REQUIRE(G(i, 0) == Catch::Approx(dN_dv_fd(i)).margin(1e-7));
            REQUIRE(G(i, 1) == Catch::Approx(dN_dw_fd(i)).margin(1e-7));
        }
    }
}

TEST_CASE("regular_basis_grad: column sums vanish (partition-of-unity invariant)",
          "[shell][loop][basis][grad][invariant]")
{
    for (const auto& s : interior_only_samples) {
        CAPTURE(s.v, s.w);
        const auto G = chladni::shell::loop::regular_basis_grad(s.v, s.w);
        REQUIRE(G.col(0).sum() == Catch::Approx(0.0).margin(1e-13));
        REQUIRE(G.col(1).sum() == Catch::Approx(0.0).margin(1e-13));
    }
}

TEST_CASE("regular_basis_hess: agrees with central second differences",
          "[shell][loop][basis][hess][fd]")
{
    // Step size for second derivatives: balance between truncation
    // error (O(h^2) for central) and round-off (O(eps/h^2)). h = 1e-3
    // gives ~1e-6 truncation and ~1e-10 round-off, comfortably matched
    // by a 1e-5 absolute tolerance.
    constexpr double h = 1.0e-3;
    for (const auto& s : interior_only_samples) {
        CAPTURE(s.v, s.w);
        const auto H = chladni::shell::loop::regular_basis_hess(s.v, s.w);
        const auto N0  = chladni::shell::loop::regular_basis(s.v, s.w);
        const auto Nvp = chladni::shell::loop::regular_basis(s.v + h, s.w);
        const auto Nvm = chladni::shell::loop::regular_basis(s.v - h, s.w);
        const auto Nwp = chladni::shell::loop::regular_basis(s.v, s.w + h);
        const auto Nwm = chladni::shell::loop::regular_basis(s.v, s.w - h);
        const auto Npp = chladni::shell::loop::regular_basis(s.v + h, s.w + h);
        const auto Nmm = chladni::shell::loop::regular_basis(s.v - h, s.w - h);
        const auto Npm = chladni::shell::loop::regular_basis(s.v + h, s.w - h);
        const auto Nmp = chladni::shell::loop::regular_basis(s.v - h, s.w + h);
        const Eigen::Matrix<double, 12, 1> d2N_dvv_fd =
            (Nvp - 2.0 * N0 + Nvm) / (h * h);
        const Eigen::Matrix<double, 12, 1> d2N_dww_fd =
            (Nwp - 2.0 * N0 + Nwm) / (h * h);
        const Eigen::Matrix<double, 12, 1> d2N_dvw_fd =
            (Npp - Npm - Nmp + Nmm) / (4.0 * h * h);
        for (Eigen::Index i = 0; i < 12; ++i) {
            CAPTURE(i);
            REQUIRE(H(i, 0) == Catch::Approx(d2N_dvv_fd(i)).margin(1e-5));
            REQUIRE(H(i, 1) == Catch::Approx(d2N_dvw_fd(i)).margin(1e-5));
            REQUIRE(H(i, 2) == Catch::Approx(d2N_dww_fd(i)).margin(1e-5));
        }
    }
}

TEST_CASE("regular_basis_hess: column sums vanish (partition-of-unity invariant)",
          "[shell][loop][basis][hess][invariant]")
{
    for (const auto& s : interior_only_samples) {
        CAPTURE(s.v, s.w);
        const auto H = chladni::shell::loop::regular_basis_hess(s.v, s.w);
        REQUIRE(H.col(0).sum() == Catch::Approx(0.0).margin(1e-13));
        REQUIRE(H.col(1).sum() == Catch::Approx(0.0).margin(1e-13));
        REQUIRE(H.col(2).sum() == Catch::Approx(0.0).margin(1e-13));
    }
}
