/**
 * @file synth.cpp
 * @brief Implementation of the resonator-bank audio synthesiser.
 *
 * Each mode is advanced by one complex multiplication per sample:
 *
 * Let @f$ z(t) = A \, e^{(-d + i\omega) t + i\varphi} @f$. Then
 * @f$ \mathrm{Im}\,z(t) = A\,e^{-d t}\,\sin(\omega t + \varphi) @f$, which
 * is exactly the per-mode contribution to the output. Stepping by one
 * sample period @f$ \Delta t = 1/f_s @f$ becomes
 * @f$ z(t + \Delta t) = z(t) \cdot e^{-d \Delta t} \cdot e^{i \omega \Delta t} @f$,
 * a single complex multiply with two precomputed scalars per mode.
 *
 * @note Magnitude drift over very long buffers is bounded by double
 *       precision and the baked-in @f$ e^{-d \Delta t} @f$ decay; a
 *       periodic renormalisation can be added later if multi-second
 *       sustains exhibit audible drift.
 */

#include <chladni/synth.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

namespace chladni::synth {

std::vector<float> synthesize_resonator_bank(
    std::span<const ResonatorMode> modes,
    double sample_rate_hz,
    double duration_s)
{
    // NaN/Inf-rejecting form: `x <= 0.0` is false for NaN, so the weaker guard
    // would let a non-finite rate/duration through — and then dt = 1/NaN = NaN
    // poisons every sample, exactly the failure the per-mode guards below reject.
    // `!(x > 0.0)` rejects NaN and non-positive; `!isfinite` also rejects ±inf.
    if (!(sample_rate_hz > 0.0) || !std::isfinite(sample_rate_hz)) {
        throw std::invalid_argument(
            "synthesize_resonator_bank: sample_rate_hz must be > 0 and finite");
    }
    if (!(duration_s > 0.0) || !std::isfinite(duration_s)) {
        throw std::invalid_argument(
            "synthesize_resonator_bank: duration_s must be > 0 and finite");
    }

    // Cap the buffer: a large-but-finite rate×duration would otherwise request
    // an absurd allocation (bad_alloc/length_error) instead of a clean domain
    // error — the wav writer caps its buffer the same way (wav.cpp). Checked on
    // the double product (finite, given the guards above) before llround, so the
    // llround/size_t cast can never see an out-of-range value.
    constexpr double kMaxSamples = 1.0e9;  // ~1.5 h at 192 kHz
    const double sample_product = sample_rate_hz * duration_s;
    if (!(sample_product <= kMaxSamples)) {
        throw std::invalid_argument(
            "synthesize_resonator_bank: sample_rate_hz × duration_s exceeds the "
            "1e9-sample ceiling");
    }
    const auto n_samples =
        static_cast<std::size_t>(std::llround(sample_product));

    // Accumulate in double, downcast on emit. Ten thousand modes summed
    // into a float would otherwise lose mantissa bits to cancellation.
    std::vector<double> acc(n_samples, 0.0);

    const double dt = 1.0 / sample_rate_hz;

    // Offline equivalent of the realtime callback's denormal floor: a
    // decaying mode's recirculating state |z| = A·exp(-d·t) eventually
    // slides into the subnormal range, where each complex multiply pays
    // the denormal-arithmetic penalty on a contribution far below the
    // float output's resolution. Rather than branch per sample, stop the
    // mode once |z| drops below this inaudible floor — its remaining
    // contribution is +0 to within float precision. (Pure throughput; the
    // output is unchanged to well past the 24-bit audio noise floor.)
    constexpr double kAmplitudeFloor = 1.0e-30;

    for (const auto& mode : modes) {
        const double omega = mode.angular_frequency;
        const double d     = mode.damping_rate;
        const double A     = mode.amplitude;
        const double phi   = mode.phase;

        // Honour the documented `damping_rate >= 0` contract: a negative
        // decay constant makes decay = exp(-d*dt) > 1, and the cutoff
        // branch below (gated on d > 0) is skipped, so the envelope grows
        // without bound and silently blows past [-1, 1]. The `!(d >= 0)`
        // form also rejects NaN.
        if (!(d >= 0.0)) {
            throw std::invalid_argument(
                "synthesize_resonator_bank: damping_rate must be >= 0 "
                "(got " + std::to_string(d) + ")");
        }
        // A non-finite omega/amplitude/phase poisons the whole output
        // buffer via acc[i] += z.imag(); reject it at the source.
        if (!std::isfinite(omega) || !std::isfinite(A)
            || !std::isfinite(phi)) {
            throw std::invalid_argument(
                "synthesize_resonator_bank: angular_frequency, amplitude, "
                "and phase must all be finite");
        }

        // z(0) = A * exp(i*phi)
        std::complex<double> z{A * std::cos(phi), A * std::sin(phi)};

        // z(t + dt) = z(t) * step   with   step = exp(-d*dt) * exp(i*omega*dt)
        const double               decay = std::exp(-d * dt);
        const std::complex<double> rot{std::cos(omega * dt),
                                       std::sin(omega * dt)};
        const std::complex<double> step = decay * rot;

        // |z(i·dt)| = |A|·decay^i drops below kAmplitudeFloor at
        // i = ln(|A|/floor) / (d·dt); clamp the loop there. d <= 0
        // (undamped) or |A| at/below the floor degenerate to the full /
        // empty ranges respectively.
        std::size_t n_active = n_samples;
        const double absA    = std::abs(A);
        if (absA <= kAmplitudeFloor) {
            n_active = 0;
        } else if (d > 0.0) {
            const double cutoff =
                std::log(absA / kAmplitudeFloor) / (d * dt);
            if (cutoff < static_cast<double>(n_samples)) {
                n_active = cutoff <= 0.0
                               ? std::size_t{0}
                               : static_cast<std::size_t>(std::ceil(cutoff));
            }
        }

        for (std::size_t i = 0; i < n_active; ++i) {
            acc[i] += z.imag();
            z *= step;
        }
    }

    std::vector<float> out(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
        out[i] = static_cast<float>(acc[i]);
    }
    return out;
}

}  // namespace chladni::synth
