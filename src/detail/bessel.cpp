/**
 * @file bessel.cpp
 * @brief Implementation of @ref include/chladni/detail/bessel.hpp.
 *
 * Each base case @f$Z_0(x), Z_1(x)@f$ uses the rational approximations
 * from Abramowitz & Stegun (1964) chapter 9, valid across the full
 * positive real line in two argument regions:
 *  - @f$x \le x^*@f$  → small-argument series in @f$(x/x^*)^2@f$.
 *  - @f$x  >  x^*@f$  → large-argument asymptotic form using auxiliary
 *                       amplitude @f$f@f$ and phase @f$\theta@f$
 *                       (J, Y), or scaled exponential (I, K).
 *
 * Higher-order @f$Z_n@f$ are obtained by forward recurrence in @f$n@f$.
 *
 * Coefficients are quoted verbatim from AS chapter 9 to keep the
 * provenance auditable; each block lists the AS equation number it
 * implements.
 */

#include <chladni/detail/bessel.hpp>

#include <cmath>
#include <numbers>

namespace chladni::detail {

namespace {

constexpr double kPi = std::numbers::pi_v<double>;

// ----- J_0 ---------------------------------------------------------------
// AS 9.4.1 (|x| <= 3) and AS 9.4.3 (x >= 3, asymptotic via f0, theta0).
double J0(double x)
{
    if (x < 3.0) {
        const double t = (x / 3.0) * (x / 3.0);
        return 1.0
             + t * (-2.2499997
             + t * ( 1.2656208
             + t * (-0.3163866
             + t * ( 0.0444479
             + t * (-0.0039444
             + t * ( 0.0002100))))));
    }
    const double t = 3.0 / x;
    const double f0 = 0.79788456
                    + t * (-0.00000077
                    + t * (-0.00552740
                    + t * (-0.00009512
                    + t * ( 0.00137237
                    + t * (-0.00072805
                    + t * ( 0.00014476))))));
    const double theta0 = x - 0.78539816
                    + t * (-0.04166397
                    + t * (-0.00003954
                    + t * ( 0.00262573
                    + t * (-0.00054125
                    + t * (-0.00029333
                    + t * ( 0.00013558))))));
    return f0 * std::cos(theta0) / std::sqrt(x);
}

// ----- J_1 ---------------------------------------------------------------
// AS 9.4.4 (|x| <= 3) and AS 9.4.6 (x >= 3).
double J1(double x)
{
    if (x < 3.0) {
        const double y = x / 3.0;
        const double t = y * y;
        const double poly = 0.5
             + t * (-0.56249985
             + t * ( 0.21093573
             + t * (-0.03954289
             + t * ( 0.00443319
             + t * (-0.00031761
             + t * ( 0.00001109))))));
        return x * poly;
    }
    const double t = 3.0 / x;
    const double f1 = 0.79788456
                    + t * ( 0.00000156
                    + t * ( 0.01659667
                    + t * ( 0.00017105
                    + t * (-0.00249511
                    + t * ( 0.00113653
                    + t * (-0.00020033))))));
    const double theta1 = x - 2.35619449
                    + t * ( 0.12499612
                    + t * ( 0.00005650
                    + t * (-0.00637879
                    + t * ( 0.00074348
                    + t * ( 0.00079824
                    + t * (-0.00029166))))));
    return f1 * std::cos(theta1) / std::sqrt(x);
}

// ----- Y_0 ---------------------------------------------------------------
// AS 9.4.2 (0 < x <= 3, with the J_0 ln(x/2) singularity factored out)
// and AS 9.4.3 (asymptotic, same f0/theta0 amplitude/phase as J_0).
double Y0(double x)
{
    if (x < 3.0) {
        const double y = x / 3.0;
        const double t = y * y;
        const double poly =  0.36746691
             + t * ( 0.60559366
             + t * (-0.74350384
             + t * ( 0.25300117
             + t * (-0.04261214
             + t * ( 0.00427916
             + t * (-0.00024846))))));
        return (2.0 / kPi) * std::log(x / 2.0) * J0(x) + poly;
    }
    const double t = 3.0 / x;
    const double f0 = 0.79788456
                    + t * (-0.00000077
                    + t * (-0.00552740
                    + t * (-0.00009512
                    + t * ( 0.00137237
                    + t * (-0.00072805
                    + t * ( 0.00014476))))));
    const double theta0 = x - 0.78539816
                    + t * (-0.04166397
                    + t * (-0.00003954
                    + t * ( 0.00262573
                    + t * (-0.00054125
                    + t * (-0.00029333
                    + t * ( 0.00013558))))));
    return f0 * std::sin(theta0) / std::sqrt(x);
}

// ----- Y_1 ---------------------------------------------------------------
// AS 9.4.5 (0 < x <= 3) and AS 9.4.6 (asymptotic, same f1/theta1 as J_1).
double Y1(double x)
{
    if (x < 3.0) {
        const double y = x / 3.0;
        const double t = y * y;
        const double poly = -0.6366198
             + t * ( 0.2212091
             + t * ( 2.1682709
             + t * (-1.3164827
             + t * ( 0.3123951
             + t * (-0.0400976
             + t * ( 0.0027873))))));
        return (2.0 / kPi) * std::log(x / 2.0) * J1(x) + poly / x;
    }
    const double t = 3.0 / x;
    const double f1 = 0.79788456
                    + t * ( 0.00000156
                    + t * ( 0.01659667
                    + t * ( 0.00017105
                    + t * (-0.00249511
                    + t * ( 0.00113653
                    + t * (-0.00020033))))));
    const double theta1 = x - 2.35619449
                    + t * ( 0.12499612
                    + t * ( 0.00005650
                    + t * (-0.00637879
                    + t * ( 0.00074348
                    + t * ( 0.00079824
                    + t * (-0.00029166))))));
    return f1 * std::sin(theta1) / std::sqrt(x);
}

// ----- I_0 ---------------------------------------------------------------
// AS 9.8.1 (|x| <= 3.75) and AS 9.8.2 (x >= 3.75, scaled by sqrt(x) e^-x).
double I0(double x)
{
    if (x < 3.75) {
        const double y = x / 3.75;
        const double t = y * y;
        return 1.0
             + t * (3.5156229
             + t * (3.0899424
             + t * (1.2067492
             + t * (0.2659732
             + t * (0.0360768
             + t * (0.0045813))))));
    }
    const double t = 3.75 / x;
    const double scaled = 0.39894228
             + t * ( 0.01328592
             + t * ( 0.00225319
             + t * (-0.00157565
             + t * ( 0.00916281
             + t * (-0.02057706
             + t * ( 0.02635537
             + t * (-0.01647633
             + t * ( 0.00392377))))))));
    return scaled * std::exp(x) / std::sqrt(x);
}

// ----- I_1 ---------------------------------------------------------------
// AS 9.8.3 (|x| <= 3.75) and AS 9.8.4 (x >= 3.75).
double I1(double x)
{
    if (x < 3.75) {
        const double y = x / 3.75;
        const double t = y * y;
        const double poly = 0.5
             + t * (0.87890594
             + t * (0.51498869
             + t * (0.15084934
             + t * (0.02658733
             + t * (0.00301532
             + t * (0.00032411))))));
        return x * poly;
    }
    const double t = 3.75 / x;
    const double scaled = 0.39894228
             + t * (-0.03988024
             + t * (-0.00362018
             + t * ( 0.00163801
             + t * (-0.01031555
             + t * ( 0.02282967
             + t * (-0.02895312
             + t * ( 0.01787654
             + t * (-0.00420059))))))));
    return scaled * std::exp(x) / std::sqrt(x);
}

// ----- K_0 ---------------------------------------------------------------
// AS 9.8.5 (0 < x <= 2) and AS 9.8.6 (x >= 2, scaled by sqrt(x) e^x).
double K0(double x)
{
    if (x <= 2.0) {
        const double y = x / 2.0;
        const double t = y * y;
        const double poly = -0.57721566
             + t * ( 0.42278420
             + t * ( 0.23069756
             + t * ( 0.03488590
             + t * ( 0.00262698
             + t * ( 0.00010750
             + t * ( 0.00000740))))));
        return -std::log(x / 2.0) * I0(x) + poly;
    }
    const double t = 2.0 / x;
    const double scaled = 1.25331414
             + t * (-0.07832358
             + t * ( 0.02189568
             + t * (-0.01062446
             + t * ( 0.00587872
             + t * (-0.00251540
             + t * ( 0.00053208))))));
    return scaled * std::exp(-x) / std::sqrt(x);
}

// ----- K_1 ---------------------------------------------------------------
// AS 9.8.7 (0 < x <= 2) and AS 9.8.8 (x >= 2).
double K1(double x)
{
    if (x <= 2.0) {
        const double y = x / 2.0;
        const double t = y * y;
        const double poly = 1.0
             + t * ( 0.15443144
             + t * (-0.67278579
             + t * (-0.18156897
             + t * (-0.01919402
             + t * (-0.00110404
             + t * (-0.00004686))))));
        return std::log(x / 2.0) * I1(x) + poly / x;
    }
    const double t = 2.0 / x;
    const double scaled = 1.25331414
             + t * ( 0.23498619
             + t * (-0.03655620
             + t * ( 0.01504268
             + t * (-0.00780353
             + t * ( 0.00325614
             + t * (-0.00068245))))));
    return scaled * std::exp(-x) / std::sqrt(x);
}

}  // anonymous namespace

double bessel_J(int n, double x)
{
    if (n == 0) return J0(x);
    if (n == 1) return J1(x);
    // Forward recurrence: J_{k+1} = (2k/x) J_k - J_{k-1}.
    double Z_prev = J0(x);
    double Z_curr = J1(x);
    for (int k = 1; k < n; ++k) {
        const double Z_next = (2.0 * k / x) * Z_curr - Z_prev;
        Z_prev = Z_curr;
        Z_curr = Z_next;
    }
    return Z_curr;
}

double bessel_Y(int n, double x)
{
    if (n == 0) return Y0(x);
    if (n == 1) return Y1(x);
    double Z_prev = Y0(x);
    double Z_curr = Y1(x);
    for (int k = 1; k < n; ++k) {
        const double Z_next = (2.0 * k / x) * Z_curr - Z_prev;
        Z_prev = Z_curr;
        Z_curr = Z_next;
    }
    return Z_curr;
}

double bessel_I(int n, double x)
{
    if (n == 0) return I0(x);
    if (n == 1) return I1(x);
    // Forward recurrence: I_{k+1} = I_{k-1} - (2k/x) I_k.
    double Z_prev = I0(x);
    double Z_curr = I1(x);
    for (int k = 1; k < n; ++k) {
        const double Z_next = Z_prev - (2.0 * k / x) * Z_curr;
        Z_prev = Z_curr;
        Z_curr = Z_next;
    }
    return Z_curr;
}

double bessel_K(int n, double x)
{
    if (n == 0) return K0(x);
    if (n == 1) return K1(x);
    // Forward recurrence: K_{k+1} = K_{k-1} + (2k/x) K_k. Stable
    // unconditionally for x > 0 (all positive terms).
    double Z_prev = K0(x);
    double Z_curr = K1(x);
    for (int k = 1; k < n; ++k) {
        const double Z_next = Z_prev + (2.0 * k / x) * Z_curr;
        Z_prev = Z_curr;
        Z_curr = Z_next;
    }
    return Z_curr;
}

double bessel_J_prime(int n, double x)
{
    if (n == 0) return -J1(x);
    return bessel_J(n - 1, x) - (static_cast<double>(n) / x) * bessel_J(n, x);
}

double bessel_Y_prime(int n, double x)
{
    if (n == 0) return -Y1(x);
    return bessel_Y(n - 1, x) - (static_cast<double>(n) / x) * bessel_Y(n, x);
}

double bessel_I_prime(int n, double x)
{
    if (n == 0) return I1(x);
    return bessel_I(n - 1, x) - (static_cast<double>(n) / x) * bessel_I(n, x);
}

double bessel_K_prime(int n, double x)
{
    if (n == 0) return -K1(x);
    return -bessel_K(n - 1, x) - (static_cast<double>(n) / x) * bessel_K(n, x);
}

}  // namespace chladni::detail
