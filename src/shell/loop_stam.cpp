/**
 * @file loop_stam.cpp
 * @brief S.1-S.3: exact-evaluation building blocks for the Stam 1999
 *        Loop-subdivision evaluator.
 *
 * Pieces:
 *  - @ref chladni::shell::loop::build_extended_subdivision_matrix /
 *    @ref chladni::shell::loop::build_extended_subdivision_matrix_bar —
 *    explicit @f$ A @f$ and @f$ \bar A @f$ from Stam App. B's blocks
 *    @f$ S, S_{11}, S_{12}, S_{21}, S_{22} @f$.
 *  - @ref chladni::shell::loop::stam_eigenstructure — analytical Fourier
 *    eigenvectors of the cyclic @f$ S @f$ block + numerical eigensolve
 *    of the constant 5x5 @f$ S_{12} @f$ + Stam Eq. (4) coupling. The
 *    @f$ N = 3 @f$ case is hardcoded from Appendix C (Jordan basis).
 *  - @ref chladni::shell::loop::stam_tile_map — Stam Sec 4 @c EvalSurf
 *    affine tile lookup.
 *  - @ref chladni::shell::loop::stam_picking_matrices — topology-driven
 *    construction of the three 12 x (N + 12) selection matrices
 *    @f$ P_1, P_2, P_3 @f$, derived by subdividing the canonical
 *    irregular patch once and reading off the box-spline-slot indices
 *    of each regular sub-patch.
 *
 * Reference: @cite stam_1999_loop_evaluation Section 3, App. B, App. C.
 */

#include <chladni/shell/loop.hpp>
#include <chladni/shell.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace chladni::shell::loop {

namespace {

constexpr double pi = std::numbers::pi_v<double>;

/// alpha(N) = 5/8 - (3 + 2 cos(2 pi / N))^2 / 64  (Stam App. B).
double alpha(int N)
{
    const double c = std::cos(2.0 * pi / static_cast<double>(N));
    const double t = 3.0 + 2.0 * c;
    return 0.625 - (t * t) / 64.0;
}

/// f(k; N) = 3/8 + (1/4) cos(2 pi k / N) — cyclic Fourier eigenvalue
/// of the cyclic 1-ring block (Stam App. B).
double f_cyclic(int N, int k)
{
    return 0.375 + 0.25 *
        std::cos(2.0 * pi * static_cast<double>(k) / static_cast<double>(N));
}

/// Build the (N+1) x (N+1) cyclic 1-ring block S (Stam App. B).
///
/// Row 0 is the extraordinary-vertex smoothing rule
/// (a_N, b_N, b_N, ..., b_N) with a_N = 1 - alpha(N), b_N = alpha(N)/N.
/// Rows 1..N are the edge-midpoint rules for the N edges
/// connecting the central vertex to its N 1-ring neighbours:
/// each new midpoint vertex k gets 3/8 from the central vertex,
/// 3/8 from its endpoint k, and 1/8 each from the previous and next
/// 1-ring neighbours (cyclic, with wrap N <-> 1).
Eigen::MatrixXd build_s(int N)
{
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N + 1, N + 1);
    const double a   = 1.0 - alpha(N);
    const double b   = alpha(N) / static_cast<double>(N);
    constexpr double c = 3.0 / 8.0;
    constexpr double d = 1.0 / 8.0;

    S(0, 0) = a;
    for (int k = 1; k <= N; ++k) {
        S(0, k) = b;
    }
    for (int k = 1; k <= N; ++k) {
        const int prev_k = (k == 1) ? N : (k - 1);
        const int next_k = (k == N) ? 1 : (k + 1);
        S(k, 0)      += c;
        S(k, k)      += c;
        S(k, prev_k) += d;
        S(k, next_k) += d;
    }
    return S;
}

/// Build the 5 x (N+1) block S_11 (Stam App. B).
///
/// Each row picks up contributions from a specific subset of the
/// central + 1-ring vertices, with row sums that — combined with the
/// 5 x 5 S_12 block — produce affine-invariant new outer-ring vertices.
Eigen::MatrixXd build_s11(int N)
{
    Eigen::MatrixXd S11 = Eigen::MatrixXd::Zero(5, N + 1);
    // Row 0: 2 6 0 ... 0 6
    S11(0, 0) = 2.0; S11(0, 1)     = 6.0; S11(0, N)     += 6.0;
    // Row 1: 1 10 1 0 ... 0 1
    S11(1, 0) = 1.0; S11(1, 1)     = 10.0; S11(1, 2)    += 1.0; S11(1, N) += 1.0;
    // Row 2: 2 6 6 0 ... 0 0
    S11(2, 0) = 2.0; S11(2, 1)     = 6.0; S11(2, 2)     += 6.0;
    // Row 3: 1 1 0 ... 0 1 10
    S11(3, 0) = 1.0; S11(3, 1)     = 1.0; S11(3, N - 1) += 1.0; S11(3, N) += 10.0;
    // Row 4: 2 0 0 ... 0 6 6
    S11(4, 0) = 2.0; S11(4, N - 1) += 6.0; S11(4, N)    += 6.0;
    return S11 / 16.0;
}

/// 5 x 5 constant block S_12 (Stam App. B). Independent of N.
Eigen::Matrix<double, 5, 5> build_s12()
{
    Eigen::Matrix<double, 5, 5> S12;
    S12 << 2, 0, 0, 0, 0,
           1, 1, 1, 0, 0,
           0, 0, 2, 0, 0,
           1, 0, 0, 1, 1,
           0, 0, 0, 0, 2;
    return S12 / 16.0;
}

/// Build the 6 x (N+1) block S_21 (Stam App. B). Each row has at most
/// 2 non-zero entries (at columns 1, 2, N-1, or N depending on row).
Eigen::MatrixXd build_s21(int N)
{
    Eigen::MatrixXd S21 = Eigen::MatrixXd::Zero(6, N + 1);
    // Row 0: 0 3 0 0 ... 0 0 1
    S21(0, 1)     = 3.0; S21(0, N)     += 1.0;
    // Row 1: 0 3 0 0 ... 0 0 0
    S21(1, 1)     = 3.0;
    // Row 2: 0 3 1 0 ... 0 0 0
    S21(2, 1)     = 3.0; S21(2, 2)     += 1.0;
    // Row 3: 0 1 0 0 ... 0 0 3
    S21(3, 1)     = 1.0; S21(3, N)     += 3.0;
    // Row 4: 0 0 0 0 ... 0 0 3
    S21(4, N)     = 3.0;
    // Row 5: 0 0 0 0 ... 0 1 3
    S21(5, N - 1) = 1.0; S21(5, N)     += 3.0;
    return S21 / 8.0;
}

/// 6 x 5 constant block S_22 (Stam App. B). Independent of N.
Eigen::Matrix<double, 6, 5> build_s22()
{
    Eigen::Matrix<double, 6, 5> S22;
    S22 << 3, 1, 0, 0, 0,
           1, 3, 1, 0, 0,
           0, 1, 3, 0, 0,
           3, 0, 0, 1, 0,
           1, 0, 0, 3, 1,
           0, 0, 0, 1, 3;
    return S22 / 8.0;
}

/// Hardcoded N=3 Jordan basis V from Stam Appendix C, column-by-column.
Eigen::Matrix<double, 9, 9> stam_v_n3()
{
    Eigen::Matrix<double, 9, 9> V;
    V <<
    //  c0  c1  c2  c3  c4  c5  c6       c7         c8
        1,  0,  0,  0,  0,  0,  0,        0.0,     33.0,
        1,  0,  1,  0,  0,  0,  0,        0.0,    -22.0,
        1, -1, -1,  0,  0,  0,  0,        0.0,    -22.0,
        1,  1,  0,  0,  0,  0,  0,        0.0,    -22.0,
        1,  3,  3,  1, -1,  0,  0,        0.0,    198.0,
        1,  0,  4,  1,  0,  0,  0, 165.0/16.0,    473.0,
        1, -3,  0,  0,  1,  0,  0,        0.0,    198.0,
        1,  4,  0,  0,  0,  1,  1, 165.0/16.0,    438.0,
        1,  0, -3, -1,  1,  1,  0,        0.0,    198.0;
    return V;
}

/// Hardcoded N=3 inverse Jordan basis V^{-1} from Stam Appendix C.
Eigen::Matrix<double, 9, 9> stam_v_inv_n3()
{
    Eigen::Matrix<double, 9, 9> Vi;
    Vi <<
        2.0/5,         1.0/5,        1.0/5,        1.0/5,         0,         0,         0,    0,    0,
        0,            -1.0/3,       -1.0/3,        2.0/3,         0,         0,         0,    0,    0,
        0,             2.0/3,       -1.0/3,       -1.0/3,         0,         0,         0,    0,    0,
       -8,             0,            3,            3,             1,         0,         1,    0,    0,
       -4,             0,            0,            3,             0,         0,         1,    0,    0,
       -8,             3,            3,            0,             1,         0,         0,    0,    1,
        7.0/11,       26.0/33,     -7.0/33,     -40.0/33,         0,        -1,         1,    1,   -1,
      -16.0/165,       0,           16.0/165,    16.0/165,    -16.0/165, 16.0/165, -16.0/165, 0,    0,
        1.0/55,       -1.0/165,    -1.0/165,    -1.0/165,         0,         0,         0,    0,    0;
    return Vi;
}

/// Hardcoded N=3 eigenvalue list (Stam App. C):
/// lambda = (1, 1/4, 1/4, 1/8, 1/8, 1/8, 1/16, 1/16, 1/16)
/// with the Jordan super-diagonal at (K-2, K-1) = (7, 8).
Eigen::VectorXd stam_lambda_n3()
{
    Eigen::VectorXd lam(9);
    lam << 1.0, 0.25, 0.25, 0.125, 0.125, 0.125,
           0.0625, 0.0625, 0.0625;
    return lam;
}

/// Build U_0, the (N+1) x (N+1) eigenvector matrix of S, plus its
/// eigenvalue diagonal Sigma_diag, using Stam's Fourier eigenanalysis
/// (App. B, real-valued form via cos/sin pairs).
///
/// Column ordering (for downstream U_1 dispatch):
///   - 0: mu_1 = 1     (constant mode (1, 1, ..., 1)^T).
///   - 1: mu_2         (extraordinary mode (-8 alpha/3, 1, 1, ..., 1)^T).
///   - 2..N: cyclic Fourier modes
///       For k = 1, ..., (N-1)/2: two real eigenvectors per k
///         (cos vector with row 1 = 1, sin vector with row 1 = 0).
///       For N even, one additional cos vector for k = N/2 at the LAST
///         column (Stam's N+1th eigenvector, with eigenvalue 1/8).
void build_u0_and_sigma(int N,
                        Eigen::MatrixXd& U0,
                        Eigen::VectorXd& Sigma_diag)
{
    U0         = Eigen::MatrixXd::Zero(N + 1, N + 1);
    Sigma_diag = Eigen::VectorXd::Zero(N + 1);
    const double a = alpha(N);

    // Col 0: eigenvalue 1, eigenvector (1, 1, ..., 1)^T.
    Sigma_diag(0) = 1.0;
    for (int i = 0; i <= N; ++i) U0(i, 0) = 1.0;

    // Col 1: eigenvalue 5/8 - alpha, eigenvector (-8 alpha/3, 1, 1, ..., 1)^T.
    Sigma_diag(1) = 0.625 - a;
    U0(0, 1) = -8.0 * a / 3.0;
    for (int i = 1; i <= N; ++i) U0(i, 1) = 1.0;

    // Cols 2..N: cyclic Fourier modes.
    int col = 2;
    for (int k = 1; k <= (N - 1) / 2; ++k) {
        const double fk          = f_cyclic(N, k);
        const double phase_unit  = 2.0 * pi * static_cast<double>(k) / static_cast<double>(N);

        // Cosine eigenvector: row 0 = 0, row 1 = 1, row j = cos((j-1)*phase) for j>=2.
        Sigma_diag(col) = fk;
        U0(1, col) = 1.0;
        for (int j = 2; j <= N; ++j) {
            U0(j, col) = std::cos(static_cast<double>(j - 1) * phase_unit);
        }
        ++col;

        // Sine eigenvector: row 0 = 0, row 1 = 0, row j = sin((j-1)*phase) for j>=2.
        Sigma_diag(col) = fk;
        for (int j = 2; j <= N; ++j) {
            U0(j, col) = std::sin(static_cast<double>(j - 1) * phase_unit);
        }
        ++col;
    }
    if (N % 2 == 0) {
        // f(N/2) = 1/8 — single cos vector (sin would be identically zero).
        const int k             = N / 2;
        const double fk         = f_cyclic(N, k);   // = 1/8
        const double phase_unit = 2.0 * pi * static_cast<double>(k) / static_cast<double>(N);  // = pi

        Sigma_diag(col) = fk;
        U0(1, col) = 1.0;
        for (int j = 2; j <= N; ++j) {
            U0(j, col) = std::cos(static_cast<double>(j - 1) * phase_unit);
        }
        ++col;
    }
    // col must equal N + 1 now.
}

}  // anonymous namespace

Eigen::MatrixXd build_extended_subdivision_matrix(int N)
{
    if (N < 3) {
        throw std::invalid_argument(
            "build_extended_subdivision_matrix: N must be >= 3 (got "
            + std::to_string(N) + ")");
    }
    const int K = N + 6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K, K);
    A.block(0,     0,     N + 1, N + 1) = build_s(N);
    A.block(N + 1, 0,     5,     N + 1) = build_s11(N);
    A.block(N + 1, N + 1, 5,     5)     = build_s12();
    return A;
}

Eigen::MatrixXd build_extended_subdivision_matrix_bar(int N)
{
    if (N < 3) {
        throw std::invalid_argument(
            "build_extended_subdivision_matrix_bar: N must be >= 3 (got "
            + std::to_string(N) + ")");
    }
    const int K = N + 6;
    const int M = N + 12;
    Eigen::MatrixXd Abar = Eigen::MatrixXd::Zero(M, K);
    // Top K rows are the unbarred A matrix.
    Abar.block(0,     0,     N + 1, N + 1) = build_s(N);
    Abar.block(N + 1, 0,     5,     N + 1) = build_s11(N);
    Abar.block(N + 1, N + 1, 5,     5)     = build_s12();
    // Bottom 6 rows: the extra rows from S_21 / S_22.
    Abar.block(N + 6, 0,     6,     N + 1) = build_s21(N);
    Abar.block(N + 6, N + 1, 6,     5)     = build_s22();
    return Abar;
}

StamEigenstructure stam_eigenstructure(int N)
{
    if (N < 3) {
        throw std::invalid_argument(
            "stam_eigenstructure: N must be >= 3 (got "
            + std::to_string(N) + ")");
    }

    // N = 3: hardcoded Jordan basis from Stam App. C.
    if (N == 3) {
        StamEigenstructure es;
        es.N                = 3;
        es.has_jordan_block = true;
        es.lambda           = stam_lambda_n3();
        es.V                = stam_v_n3();
        es.V_inv            = stam_v_inv_n3();
        return es;
    }

    const int K = N + 6;

    // --- Top-left S block: analytical Fourier eigenstructure. ---
    Eigen::MatrixXd U0;
    Eigen::VectorXd Sigma_diag;
    build_u0_and_sigma(N, U0, Sigma_diag);

    // --- Bottom-right S_12 block: numerical eigendecomposition. ---
    // S_12 is 5x5, constant. EigenSolver gives complex eigenvalues,
    // but S_12's eigenvalues are real (1/8 mult 3, 1/16 mult 2) and
    // it is diagonalizable, so the real parts of the complex output
    // are the true eigenstructure (imag parts are < 1e-15 noise).
    const Eigen::Matrix<double, 5, 5> S12 = build_s12();
    Eigen::EigenSolver<Eigen::Matrix<double, 5, 5>> es12(S12, /*computeEigenvectors=*/true);
    Eigen::VectorXd Delta(5);
    Eigen::MatrixXd W1(5, 5);
    for (int i = 0; i < 5; ++i) {
        Delta(i) = es12.eigenvalues()(i).real();
        for (int j = 0; j < 5; ++j) {
            W1(j, i) = es12.eigenvectors()(j, i).real();
        }
    }

    // --- Off-diagonal coupling U_1: solve (sigma I - S_12) u_1 = S_11 u_0. ---
    // Special-case sigma = 1/8 (the f(N/2) column for N even): Stam's
    // explicit solution u_1 = (0, 8, 0, -8, 0)^T. (sigma = 1/16 collisions
    // would need a Jordan basis, but they only occur at N = 3 — handled
    // by the early-return above.)
    const Eigen::MatrixXd S11 = build_s11(N);
    const Eigen::Matrix<double, 5, 5> I5 = Eigen::Matrix<double, 5, 5>::Identity();

    Eigen::MatrixXd U1(5, N + 1);
    for (int i = 0; i < N + 1; ++i) {
        const double sigma = Sigma_diag(i);
        const Eigen::VectorXd u0_i = U0.col(i);

        if (std::abs(sigma - 0.125) < 1.0e-12) {
            U1.col(i) << 0.0, 8.0, 0.0, -8.0, 0.0;
        } else {
            const Eigen::Matrix<double, 5, 5> M = sigma * I5 - S12;
            U1.col(i) = M.fullPivLu().solve(S11 * u0_i);
        }
    }

    // --- Assemble V = [U_0 0; U_1 W_1] and lambda. ---
    Eigen::MatrixXd V        = Eigen::MatrixXd::Zero(K, K);
    V.block(0,     0,     N + 1, N + 1) = U0;
    V.block(N + 1, 0,     5,     N + 1) = U1;
    V.block(N + 1, N + 1, 5,     5)     = W1;

    Eigen::VectorXd lambda(K);
    lambda.head(N + 1) = Sigma_diag;
    lambda.tail(5)     = Delta;

    StamEigenstructure es;
    es.N                = N;
    es.has_jordan_block = false;
    es.lambda           = lambda;
    es.V                = V;
    es.V_inv            = V.inverse();
    return es;
}

namespace {

/// Build the canonical irregular patch around an extraordinary vertex of
/// valence @c N: K = N + 6 control vertices and N + 7 CCW manifold faces.
///
/// Vertex labelling (matching @ref build_extended_subdivision_matrix_bar):
///   0       : extraordinary vertex (at the origin, valence N)
///   1..N    : 1-ring, placed CW from above so that vertex 1 sits at
///             angle 0 and vertex N at angle +2π/N (after mod 2π); the
///             central spoke (0, 1, N) is then CCW from above.
///   N+1     : outer vertex across edge (1, N) from vertex 0, at angle
///             +π/N, radius 2.
///   N+2     : next outer around vertex 1 CW from N+1, at angle -π/(2N).
///   N+3     : outer across edge (1, 2), at angle -π/N.
///   N+4     : next outer around vertex N CCW from N+1, at angle
///             +2π/N + π/(2N).
///   N+5     : outer across edge (N, N-1), at angle +3π/N.
///
/// Faces (all CCW from +z; F.row(0) is the central triangle):
///   row 0          : (0, 1, N)                — central spoke
///   rows 1..N-1    : (0, k+1, k) for k=1..N-1 — remaining N-1 spokes
///   rows N+0..N+3  : the 4 outer faces fanning around vertex 1
///   rows N+4..N+6  : the 3 outer faces fanning around vertex N
///
/// Vertex positions are chosen for diagnostic convenience; only the
/// patch topology matters for picking-matrix construction.
void build_irregular_patch(int N,
                           Eigen::MatrixXd& V,
                           Eigen::MatrixXi& F)
{
    const int K = N + 6;
    V.resize(K, 3);

    V.row(0).setZero();
    for (int k = 1; k <= N; ++k) {
        const double angle = -2.0 * pi * static_cast<double>(k - 1)
                                       / static_cast<double>(N);
        V.row(k) << std::cos(angle), std::sin(angle), 0.0;
    }
    constexpr double R_outer = 2.0;
    auto place = [&](int idx, double angle) {
        V.row(idx) << R_outer * std::cos(angle),
                      R_outer * std::sin(angle),
                      0.0;
    };
    const double Nf = static_cast<double>(N);
    place(N + 1,  pi / Nf);
    place(N + 2, -pi / (2.0 * Nf));
    place(N + 3, -pi / Nf);
    place(N + 4,  2.0 * pi / Nf + pi / (2.0 * Nf));
    place(N + 5,  3.0 * pi / Nf);

    F.resize(N + 7, 3);
    F.row(0) << 0, 1, N;
    for (int k = 1; k <= N - 1; ++k) {
        F.row(k) << 0, k + 1, k;
    }
    F.row(N + 0) << 1,     N + 1, N;
    F.row(N + 1) << 1,     N + 2, N + 1;
    F.row(N + 2) << 1,     N + 3, N + 2;
    F.row(N + 3) << 1,     2,     N + 3;
    F.row(N + 4) << N,     N + 1, N + 4;
    F.row(N + 5) << N,     N + 4, N + 5;
    F.row(N + 6) << N,     N + 5, N - 1;
}

/// Locate the index of the edge with endpoints (a, b) in @p edges
/// (@ref chladni::shell::build_edges output, sorted by (v0, v1)).
/// Returns -1 if not found.
Eigen::Index find_edge_index(
    const std::vector<chladni::shell::Edge>& edges,
    Eigen::Index a, Eigen::Index b)
{
    if (a > b) std::swap(a, b);
    for (Eigen::Index e = 0;
         e < static_cast<Eigen::Index>(edges.size()); ++e)
    {
        const auto& edge = edges[static_cast<std::size_t>(e)];
        if (edge.v0 == a && edge.v1 == b) return e;
    }
    return -1;
}

/// Build the V_sub-index -> Stam-M-index lookup for the subdivided
/// canonical patch. Returns a length-@c n_sub vector with @c -1 for
/// V_sub vertices not in Stam's M = N + 12 set.
///
/// The Stam-M correspondence (from @ref build_extended_subdivision_matrix_bar):
///  - Top S block (rows 0..N): smoothed centre + N central-spoke midpoints.
///  - Middle [S_11 S_12] block (rows N+1..N+5): m_{1,N}, smoothed v1,
///    m_{1,2}, smoothed v_N, m_{N-1,N}.
///  - Bottom [S_21 S_22] block (rows N+6..N+11): m_{1,N+1}, m_{1,N+2},
///    m_{1,N+3}, m_{N,N+1}, m_{N,N+4}, m_{N,N+5}.
std::vector<int> build_vsub_to_stam_map(
    int N,
    const std::vector<chladni::shell::Edge>& edges,
    Eigen::Index n_sub)
{
    const int K = N + 6;
    std::vector<int> map(static_cast<std::size_t>(n_sub), -1);

    // Top S block — even rule on vertex 0, plus midpoints (0, k).
    map[0] = 0;                                  // smoothed extraordinary
    for (int k = 1; k <= N; ++k) {
        const Eigen::Index e = find_edge_index(edges, 0, k);
        if (e >= 0) map[static_cast<std::size_t>(K + e)] = k;
    }

    // Middle [S_11 S_12] block.
    map[1] = N + 2;                                                // smoothed v1
    map[static_cast<std::size_t>(N)] = N + 4;                      // smoothed v_N
    {
        const Eigen::Index e = find_edge_index(edges, 1, N);
        if (e >= 0) map[static_cast<std::size_t>(K + e)] = N + 1;
    }
    {
        const Eigen::Index e = find_edge_index(edges, 1, 2);
        if (e >= 0) map[static_cast<std::size_t>(K + e)] = N + 3;
    }
    {
        const Eigen::Index e = find_edge_index(edges, N - 1, N);
        if (e >= 0) map[static_cast<std::size_t>(K + e)] = N + 5;
    }

    // Bottom [S_21 S_22] block — six outer-extension midpoints.
    const std::array<std::pair<int, int>, 6> outer_edges = {{
        {1, N + 1}, {1, N + 2}, {1, N + 3},
        {N, N + 1}, {N, N + 4}, {N, N + 5}
    }};
    for (int i = 0; i < 6; ++i) {
        const Eigen::Index e = find_edge_index(
            edges, outer_edges[static_cast<std::size_t>(i)].first,
                   outer_edges[static_cast<std::size_t>(i)].second);
        if (e >= 0) map[static_cast<std::size_t>(K + e)] = N + 6 + i;
    }
    return map;
}

}  // anonymous namespace

StamTileMap stam_tile_map(double v, double w)
{
    if (v < 0.0 || w < 0.0) {
        throw std::invalid_argument(
            "stam_tile_map: (v, w) must be non-negative (got v="
            + std::to_string(v) + ", w=" + std::to_string(w) + ")");
    }
    constexpr double kSumTol = 1.0e-12;
    if (v + w > 1.0 + kSumTol) {
        throw std::invalid_argument(
            "stam_tile_map: (v, w) outside unit triangle, v + w = "
            + std::to_string(v + w) + " > 1");
    }
    // Degenerate at the extraordinary vertex (v + w -> 0): no finite
    // tile contains the origin. log2(0) would also underflow.
    constexpr double kMinSum = 1.0e-300;
    if (v + w < kMinSum) {
        throw std::invalid_argument(
            "stam_tile_map: (v, w) at the extraordinary-vertex limit "
            "(v + w = 0); no finite tile applies");
    }

    const double sum = v + w;
    int n = static_cast<int>(std::floor(1.0 - std::log2(sum)));
    // sum == 1 -> log2 = 0 -> n = 1. Round-off near sum = 1 can shave
    // n down to 0; clamp.
    if (n < 1) n = 1;
    // Cap n: the kMinSum guard only rejects sum == 0, but a tiny-but-nonzero
    // sum (e.g. ~1e-300 from a custom near-apex quadrature rule) drives
    // n ~ 1000, so pow(2, n-1) overflows to +inf and the rescaled tile
    // parameters become non-finite. The Loop eigenstructure decays as
    // (5/8)^{n-1}, below double precision past n ~ 60, so any tile this deep
    // is numerically indistinguishable from the apex; clamp to keep the
    // public-API output finite for callers with custom rules (review7 I8).
    // Shipped quadrature rules sample sum >~ 0.1 (n <= ~4), so this never
    // fires on the assembly path.
    if (n > 60) n = 60;

    const double pow2_nm1 = std::pow(2.0, n - 1);
    const double v_s = v * pow2_nm1;
    const double w_s = w * pow2_nm1;

    StamTileMap tm;
    tm.n = n;
    if (v_s > 0.5) {
        tm.k   = 1;
        tm.v_p = 2.0 * v_s - 1.0;
        tm.w_p = 2.0 * w_s;
    } else if (w_s > 0.5) {
        tm.k   = 3;
        tm.v_p = 2.0 * v_s;
        tm.w_p = 2.0 * w_s - 1.0;
    } else {
        tm.k   = 2;
        tm.v_p = 1.0 - 2.0 * v_s;
        tm.w_p = 1.0 - 2.0 * w_s;
    }
    return tm;
}

StamPickingMatrices stam_picking_matrices(int N)
{
    if (N < 3) {
        throw std::invalid_argument(
            "stam_picking_matrices: N must be >= 3 (got "
            + std::to_string(N) + ")");
    }
    const int M = N + 12;

    // Build the canonical irregular patch and apply one Loop subdivision
    // step. Vertex positions are diagnostic only; topology is what
    // matters here.
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    build_irregular_patch(N, V, F);

    const auto subdiv   = loop_subdivide_one_step(V, F);
    const auto edges    = chladni::shell::build_edges(F);
    const auto stencils = build_patch_stencils(subdiv.V_sub, subdiv.F_sub);

    const auto vsub_to_stam = build_vsub_to_stam_map(
        N, edges, subdiv.V_sub.rows());

    // F_sub.row(0..3) for face 0 = central triangle (a=0, b=1, c=N):
    //   row 0: (0, m_{0,1}, m_{0,N})       — irregular (corner at extraordinary)
    //   row 1: (m_{0,1}, 1, m_{1,N})       — regular  → tile k = 1 (corner at v=1)
    //   row 2: (m_{0,N}, m_{1,N}, N)       — regular  → tile k = 3 (corner at w=1)
    //   row 3: (m_{0,1}, m_{1,N}, m_{0,N}) — medial   → tile k = 2
    constexpr std::array<Eigen::Index, 3> sub_face_idx = {1, 3, 2};

    StamPickingMatrices out;
    out.N = N;
    for (int k = 0; k < 3; ++k) {
        const auto canonical_dofs = canonical_regular_dofs(
            stencils[static_cast<std::size_t>(
                sub_face_idx[static_cast<std::size_t>(k)])],
            subdiv.F_sub);

        Eigen::Matrix<double, 12, Eigen::Dynamic> P
            = Eigen::Matrix<double, 12, Eigen::Dynamic>::Zero(12, M);
        for (int slot = 0; slot < 12; ++slot) {
            const Eigen::Index v_sub = canonical_dofs[
                static_cast<std::size_t>(slot)];
            const int stam_idx = vsub_to_stam[
                static_cast<std::size_t>(v_sub)];
            if (stam_idx < 0) {
                throw std::runtime_error(
                    "stam_picking_matrices: canonical DOF "
                    + std::to_string(static_cast<long long>(v_sub))
                    + " (tile k=" + std::to_string(k + 1)
                    + ", slot=" + std::to_string(slot)
                    + ") is not in Stam's M=N+12 set for N="
                    + std::to_string(N)
                    + " (programmer error)");
            }
            P(slot, stam_idx) = 1.0;
        }
        out.P[static_cast<std::size_t>(k)] = std::move(P);
    }
    return out;
}

StamEvaluator make_stam_evaluator(int N)
{
    if (N < 3) {
        throw std::invalid_argument(
            "make_stam_evaluator: N must be >= 3 (got "
            + std::to_string(N) + ")");
    }
    StamEvaluator ev;
    ev.N = N;

    const auto eig   = stam_eigenstructure(N);
    const auto Abar  = build_extended_subdivision_matrix_bar(N);
    const auto picks = stam_picking_matrices(N);

    ev.has_jordan_block = eig.has_jordan_block;
    ev.lambda           = eig.lambda;
    ev.V_inv_T          = eig.V_inv.transpose();

    // Cache (P_k * Abar * V)^T as a K x 12 matrix per tile.
    for (int k = 0; k < 3; ++k) {
        const Eigen::Matrix<double, 12, Eigen::Dynamic> Pk_Abar_V
            = picks.P[static_cast<std::size_t>(k)] * Abar * eig.V;
        ev.Pk_Abar_V_T[static_cast<std::size_t>(k)] = Pk_Abar_V.transpose();
    }
    return ev;
}

namespace {

/// Per-tile sign for the gradient chain rule:
///   sigma_1 = +1 (corner-1 sub-tile)
///   sigma_2 = -1 (medial sub-tile is flipped, t_{n,2}(v, w) =
///                 (1 - 2^n v, 1 - 2^n w))
///   sigma_3 = +1 (corner-3 sub-tile)
double tile_grad_sign(int k) noexcept
{
    return (k == 2) ? -1.0 : 1.0;
}

/// Apply Lambda^{n-1} as a row-wise scaling of an X x cols matrix in
/// place, then — for N == 3 (@p has_jordan) — add the Stam App. C
/// Jordan-block correction on top.
///
/// Stam Eq. (5) is `Phi = Lambda^{n-1} (P_k Abar V)^T b`. Tracing back
/// to Stam Eq. (2) `s = b^T P_k Abar A^{n-1} C_0` with `A = V J V^{-1}`
/// (J = our standard super-diagonal Jordan with `J(K-2, K-1) = 1`),
/// matching coefficients on `b_l Q(l, j) ... C_hat(i)` shows that
/// Stam's "Lambda^{n-1}" actually equals `(J^T)^{n-1}`. So the
/// (n-1) lambda^(n-2) off-diagonal contribution lives at
/// `(J^T)^{n-1}(K-1, K-2)` — the *sub*-diagonal — and lifts
/// `Phi(K-1)` (NOT `Phi(K-2)`) by `(n-1) lambda^(n-2)` times the
/// pre-scaling `u(K-2)` row. See Stam Appendix C.
///
/// The pre-scaling row at index K-2 is captured up front so the order
/// of the two passes is well-defined. When @c n_minus_1 == 0, both
/// passes degenerate to no-ops (Lambda^0 = I and the Jordan term
/// carries an (n-1) = 0 coefficient).
void apply_lambda_pow_and_jordan_in_place(
    const Eigen::VectorXd&         lambda,
    bool                           has_jordan,
    int                            n_minus_1,
    Eigen::Ref<Eigen::MatrixXd>    M)
{
    if (n_minus_1 == 0) return;
    const Eigen::Index K = lambda.size();

    // Save M.row(K-2) BEFORE Lambda^{n-1} scaling, so the Jordan
    // correction sees the unscaled u row.
    Eigen::RowVectorXd u_jord_src;
    if (has_jordan) {
        u_jord_src = M.row(K - 2);
    }

    for (Eigen::Index i = 0; i < K; ++i) {
        const double lam_pow = std::pow(lambda(i), n_minus_1);
        M.row(i) *= lam_pow;
    }

    if (has_jordan) {
        const double lambda_jord = lambda(K - 2);
        const double jord_coef = static_cast<double>(n_minus_1)
                               * std::pow(lambda_jord, n_minus_1 - 1);
        M.row(K - 1) += jord_coef * u_jord_src;
    }
}

}  // anonymous namespace

Eigen::VectorXd stam_phi(const StamEvaluator& ev, double v, double w)
{
    // Tile + parameter remap (Stam Sec 4 EvalSurf).
    const StamTileMap tm = stam_tile_map(v, w);

    // Regular box-spline basis at the remapped (v_p, w_p) on the unit
    // triangle of the regular sub-tile.
    const Eigen::Matrix<double, 12, 1> b = regular_basis(tm.v_p, tm.w_p);

    // Phi = J^{n-1} (P_k Abar V)^T b. Stam Eq. (5).
    Eigen::MatrixXd phi
        = ev.Pk_Abar_V_T[static_cast<std::size_t>(tm.k - 1)] * b;
    apply_lambda_pow_and_jordan_in_place(
        ev.lambda, ev.has_jordan_block, tm.n - 1, phi);
    return phi.col(0);
}

Eigen::Matrix<double, Eigen::Dynamic, 2>
stam_phi_grad(const StamEvaluator& ev, double v, double w)
{
    const StamTileMap tm = stam_tile_map(v, w);
    const Eigen::Matrix<double, 12, 2> bg
        = regular_basis_grad(tm.v_p, tm.w_p);

    // Chain-rule factor: sigma_k * 2^n. Diagonal Jacobian, so v- and
    // w-partials independently pick up the same scalar factor.
    const double factor = tile_grad_sign(tm.k) * std::pow(2.0, tm.n);

    Eigen::MatrixXd phi_grad
        = ev.Pk_Abar_V_T[static_cast<std::size_t>(tm.k - 1)] * bg;
    phi_grad *= factor;

    apply_lambda_pow_and_jordan_in_place(
        ev.lambda, ev.has_jordan_block, tm.n - 1, phi_grad);
    return phi_grad;
}

Eigen::Matrix<double, Eigen::Dynamic, 3>
stam_phi_hess(const StamEvaluator& ev, double v, double w)
{
    const StamTileMap tm = stam_tile_map(v, w);
    const Eigen::Matrix<double, 12, 3> bh
        = regular_basis_hess(tm.v_p, tm.w_p);

    // Chain-rule factor: sigma_k^2 * 4^n = 4^n (no per-tile sign for
    // second derivatives — the squared Jacobian and the sigma * sigma
    // cross term both come out positive for all 3 tiles).
    const double factor = std::pow(4.0, tm.n);

    Eigen::MatrixXd phi_hess
        = ev.Pk_Abar_V_T[static_cast<std::size_t>(tm.k - 1)] * bh;
    phi_hess *= factor;

    apply_lambda_pow_and_jordan_in_place(
        ev.lambda, ev.has_jordan_block, tm.n - 1, phi_hess);
    return phi_hess;
}

StamPatchEvaluation evaluate_patch_stam(
    const StamEvaluator&             ev,
    const std::vector<Eigen::Index>& patch_dofs,
    const Eigen::MatrixXd&           V,
    double                           v,
    double                           w)
{
    if (V.cols() < 3) {
        throw std::invalid_argument(
            "evaluate_patch_stam: V must have at least 3 columns");
    }
    const Eigen::Index K = static_cast<Eigen::Index>(ev.N) + 6;
    if (static_cast<Eigen::Index>(patch_dofs.size()) != K) {
        throw std::invalid_argument(
            "evaluate_patch_stam: patch_dofs.size() must be N + 6 = "
            + std::to_string(static_cast<long long>(K)) + " (got "
            + std::to_string(patch_dofs.size()) + ")");
    }

    StamPatchEvaluation pe;
    // Stam's Phi(v, w) is the EIGENBASIS (K-vec). To get the NODE
    // BASIS eta — the partition-of-unity functions indexed by control
    // vertex — we apply V^{-T}: eta = V^{-T} Phi (Stam Eq. 6 written
    // surface-side as s = C_0^T eta with eta = V^{-T} Phi). The
    // per-element kernel below contracts eta against patch vertex
    // positions, so it must see the node basis, not the eigenbasis.
    const Eigen::VectorXd phi = stam_phi(ev, v, w);
    const Eigen::Matrix<double, Eigen::Dynamic, 2> phi_grad
        = stam_phi_grad(ev, v, w);
    const Eigen::Matrix<double, Eigen::Dynamic, 3> phi_hess
        = stam_phi_hess(ev, v, w);
    pe.N      = ev.V_inv_T * phi;            // K-vec
    pe.N_grad = ev.V_inv_T * phi_grad;       // K x 2
    pe.N_hess = ev.V_inv_T * phi_hess;       // K x 3

    // Gather the K control points into a K x 3 matrix.
    Eigen::MatrixXd P(K, 3);
    for (Eigen::Index i = 0; i < K; ++i) {
        P.row(i) = V.row(patch_dofs[static_cast<std::size_t>(i)])
                       .head<3>();
    }

    // 3D quantities: position = sum_I N_I * P_I, etc.
    pe.position      = (P.transpose() * pe.N);
    pe.cov_basis     = P.transpose() * pe.N_grad;
    pe.second_derivs = P.transpose() * pe.N_hess;

    pe.normal = pe.cov_basis.col(0).cross(pe.cov_basis.col(1)).normalized();
    return pe;
}

namespace {

// Accumulate per-quadrature-point K contribution at a Stam patch.
// Stam K size is dynamic (N+6), so we don't use the templated helper
// from loop.cpp; structure mirrors it.
void accumulate_stiffness_stam_at_point(
    const StamPatchEvaluation& pe,
    double                     w_q,
    const chladni::shell::ShellMaterial& material,
    Eigen::MatrixXd&           K_e)
{
    const Eigen::Index K  = pe.N.size();

    const Eigen::Vector3d a1  = pe.cov_basis.col(0);
    const Eigen::Vector3d a2  = pe.cov_basis.col(1);
    const Eigen::Vector3d a3  = pe.normal;
    const Eigen::Vector3d a11 = pe.second_derivs.col(0);
    const Eigen::Vector3d a12 = pe.second_derivs.col(1);
    const Eigen::Vector3d a22 = pe.second_derivs.col(2);

    const double cov11 = a1.dot(a1);
    const double cov12 = a1.dot(a2);
    const double cov22 = a2.dot(a2);
    const double a_det = cov11 * cov22 - cov12 * cov12;
    // A non-finite a_det fails `a_det <= 0.0` (NaN compares false), so
    // guard finiteness explicitly — otherwise sqrt(NaN) propagates into
    // K_e. Mirrors the mass kernel's `!isfinite || <= 0` guard.
    if (!std::isfinite(a_det) || a_det <= 0.0) {
        throw std::runtime_error(
            "element_stiffness_stam: degenerate parametrisation "
            "(|a_1 x a_2|^2 = " + std::to_string(a_det) + ")");
    }
    const double sqrt_a    = std::sqrt(a_det);
    const double inv_det   = 1.0 / a_det;
    const double inv_sqrt_a = 1.0 / sqrt_a;

    const double con11 =  cov22 * inv_det;
    const double con22 =  cov11 * inv_det;
    const double con12 = -cov12 * inv_det;

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

    const Eigen::Index dim = 3 * K;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3, dim);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, dim);

    const Eigen::Vector3d a2xa3 = a2.cross(a3);
    const Eigen::Vector3d a3xa1 = a3.cross(a1);

    for (Eigen::Index I = 0; I < K; ++I) {
        const double dN_dv    = pe.N_grad(I, 0);
        const double dN_dw    = pe.N_grad(I, 1);
        const double d2N_dv2  = pe.N_hess(I, 0);
        const double d2N_dvdw = pe.N_hess(I, 1);
        const double d2N_dw2  = pe.N_hess(I, 2);

        M.block<3, 3>(0, 3 * I).row(0) = dN_dv * a1.transpose();
        M.block<3, 3>(0, 3 * I).row(1) = dN_dw * a2.transpose();
        M.block<3, 3>(0, 3 * I).row(2) =
            (dN_dw * a1 + dN_dv * a2).transpose();

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

        B.block<3, 3>(0, 3 * I).row(0) = B_I_1.transpose();
        B.block<3, 3>(0, 3 * I).row(1) = B_I_2.transpose();
        B.block<3, 3>(0, 3 * I).row(2) = 2.0 * B_I_3.transpose();
    }

    const double scale = w_q * sqrt_a;
    K_e.noalias() += scale * material.k_L * (M.transpose() * H * M);
    K_e.noalias() += scale * material.k_B * (B.transpose() * H * B);
}

}  // namespace

Eigen::MatrixXd element_stiffness_stam(
    const StamEvaluator&                  ev,
    const std::vector<Eigen::Index>&      patch_dofs,
    const Eigen::MatrixXd&                V_aug,
    const chladni::shell::ShellMaterial&  material,
    QuadratureRule                        rule)
{
    // Reference-triangle quadrature. Defaults to SevenPointDunavant —
    // matches the regular K and Stam M kernels so K and M live on the
    // same rule and the generalized eigenproblem is well-posed.
    //
    // R12 (review 2026-06-09): a SINGLE rule over the whole reference
    // triangle is not exact for the Stam patch — the limit basis is
    // piecewise-polynomial across the dyadic tiles Omega_k^n shrinking onto
    // the extraordinary vertex, and Dunavant's innermost point (0.10, 0.10)
    // sits at tile level ~3, missing every level >= 4. The exact cure is
    // per-tile integration (partition into n_rings dyadic rings, Dunavant on
    // each tile). Measured under-integration of the basis-gradient energy:
    // ~0.96 % at valence 3, ~0.29-0.40 % at valence 5/6/7, converging
    // geometrically (n_rings=4 -> ~1e-4). Because irregular patches are a
    // small minority of any mesh, the GLOBAL modal impact is well below the
    // validated ~0.03 % icosphere accuracy, so the single rule is kept; the
    // per-tile rule is not worth its ~12x per-patch cost here. See the
    // measurement in tests/shell/test_stam_quadrature_spike.cpp
    // ([stam_quad_spike], STAM_DIAG=1) — the standing evidence and the
    // ready-made subdivided-rule generator if exactness is ever needed.
    const auto qp = quadrature_points(rule);

    const Eigen::Index K = static_cast<Eigen::Index>(patch_dofs.size());
    const Eigen::Index dim = 3 * K;
    Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(dim, dim);

    for (const auto& q : qp) {
        const auto pe = evaluate_patch_stam(ev, patch_dofs, V_aug, q.v, q.w);
        accumulate_stiffness_stam_at_point(pe, q.weight, material, K_e);
    }
    return K_e;
}

// ---------------------------------------------------------------------------
// Per-element M for an irregular Stam patch (7-point Dunavant degree-5).
// ---------------------------------------------------------------------------

Eigen::MatrixXd element_mass_stam(
    const StamEvaluator&             ev,
    const std::vector<Eigen::Index>& patch_dofs,
    const Eigen::MatrixXd&           V_aug,
    double                           surface_density,
    QuadratureRule                   rule)
{
    // Reference-triangle quadrature. Default SevenPointDunavant
    // matches element_mass_regular. The Stam basis is non-singular
    // everywhere on the unit triangle except the apex (0, 0); all
    // sample points of the supported rules are safely inside the
    // well-conditioned region (closest is (1/3, 1/3) for OnePointCentroid
    // and (beta_1, beta_1) ≈ (0.10, 0.10) for SevenPointDunavant).
    const auto qp = quadrature_points(rule);

    const Eigen::Index K = static_cast<Eigen::Index>(patch_dofs.size());
    if (K <= 0) {
        throw std::invalid_argument(
            "element_mass_stam: empty patch_dofs");
    }
    Eigen::MatrixXd Me_scalar = Eigen::MatrixXd::Zero(K, K);

    for (const auto& q : qp) {
        const double v   = q.v;
        const double w   = q.w;
        const double w_q = q.weight;

        const auto pe = evaluate_patch_stam(ev, patch_dofs, V_aug, v, w);

        const Eigen::Vector3d a1 = pe.cov_basis.col(0);
        const Eigen::Vector3d a2 = pe.cov_basis.col(1);
        const double sqrt_a = a1.cross(a2).norm();
        if (!std::isfinite(sqrt_a) || sqrt_a <= 0.0) {
            throw std::runtime_error(
                "element_mass_stam: degenerate parametrisation at "
                "(v,w)=(" + std::to_string(v) + "," + std::to_string(w)
                + "), |a1xa2| = " + std::to_string(sqrt_a));
        }

        const double scale = w_q * surface_density * sqrt_a;
        Me_scalar.noalias() += scale * (pe.N * pe.N.transpose());
    }

    // Inflate K x K scalar mass to 3K x 3K with 3x3 identity blocks.
    const Eigen::Index dim = 3 * K;
    Eigen::MatrixXd Me = Eigen::MatrixXd::Zero(dim, dim);
    for (Eigen::Index I = 0; I < K; ++I) {
        for (Eigen::Index J = 0; J < K; ++J) {
            const double m_IJ = Me_scalar(I, J);
            for (int d = 0; d < 3; ++d) {
                Me(3 * I + d, 3 * J + d) = m_IJ;
            }
        }
    }
    return Me;
}

}  // namespace chladni::shell::loop
