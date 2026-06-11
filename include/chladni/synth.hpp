#pragma once

/**
 * @file synth.hpp
 * @brief Mono audio synthesis from a bank of damped sinusoids (resonator bank).
 *
 * This is the audio backend used by the strike pipeline:
 *
 * @verbatim
 *     impulse f
 *         |
 *         v
 *     modal projection a_i = phi_i^T f                (chladni::modes)
 *         |
 *         v
 *     ResonatorMode { omega_i, d_i, a_i }             (this header)
 *         |
 *         v
 *     synthesize_resonator_bank(...) -> float buffer  (this header)
 *         |
 *         v
 *     miniaudio output device   /   chladni::wav writer
 * @endverbatim
 *
 * It deliberately knows nothing about meshes, FEM, or material parameters
 * so that it can be developed and tested independently of the modal solver
 * (synthesise hand-picked spectra, listen, iterate). The hand-off contract
 * between solver and synthesiser is just `std::span<const ResonatorMode>`.
 *
 * @section synth_refs References
 * - @cite vandendoel_1998_physical_shapes — modal sound foundations.
 * - @cite obrien_2002_rigid_body — modal sound from rigid-body simulation.
 * - @cite bonneel_2008_fast_modal — frequency-domain optimisation
 *   (relevant when the bank exceeds a few hundred modes).
 */

#include <cstddef>
#include <span>
#include <vector>

namespace chladni::synth {

/**
 * @brief One resonator (damped sinusoidal partial).
 *
 * A single mode contributes to the output buffer the time-series
 * @f$ s_i(t) \;=\; A_i \, e^{-d_i t} \, \sin(\omega_i t + \varphi_i) @f$,
 * for @f$ t \ge 0 @f$. The full output is the linear sum over modes.
 *
 * @note `damping_rate` is the per-mode envelope decay constant, in 1/s.
 *       Converting from a perceptual T60 (time to attenuate by 60 dB) is
 *       @f$ d = \dfrac{\ln(10^{3})}{T_{60}} = \dfrac{6.9078}{T_{60}} @f$.
 *       Converting from a quality factor is @f$ d = \omega / (2 Q) @f$.
 */
struct ResonatorMode {
    double angular_frequency;  ///< @f$\omega@f$, mode angular frequency (rad/s).
    double damping_rate;       ///< @f$d@f$, exponential decay constant (1/s), @f$\ge 0@f$.
    double amplitude;          ///< @f$A@f$, initial linear amplitude (dimensionless).
    double phase = 0.0;        ///< @f$\varphi@f$, sine phase offset (rad).
                               ///< Defaults to 0 (an impulse-driven mode).
};

/**
 * @brief Synthesize a mono float32 audio buffer from a bank of resonators.
 *
 * Each output sample at time @f$ t = i / f_s @f$ (with @f$ f_s @f$ the sample
 * rate) is
 * @f[
 *   y[i] \;=\; \sum_{k} A_k\,e^{-d_k t}\,\sin(\omega_k t + \varphi_k).
 * @f]
 *
 * The function does NOT normalise or clip its output. Callers that send
 * the buffer to an audio device or to @ref chladni::wav writers are
 * responsible for ensuring the result lies in @f$[-1, 1]@f$ (typically by
 * scaling the modal amplitudes after projection).
 *
 * @param modes           Resonator bank to sum. Empty span yields a buffer
 *                        of zeros of the requested length.
 * @param sample_rate_hz  Sample rate of the output buffer in Hz; @f$> 0@f$.
 * @param duration_s      Duration of the output buffer in seconds; @f$> 0@f$.
 *
 * @return A `std::vector<float>` of length
 *         @f$ N = \mathrm{round}(\text{sample\_rate\_hz} \cdot \text{duration\_s}) @f$,
 *         containing the summed time series. The first sample corresponds
 *         to @f$ t = 0 @f$.
 *
 * @throws std::invalid_argument if `sample_rate_hz` or `duration_s` is
 *         non-positive.
 *
 * @note The reference implementation steps each mode forward by one
 *       complex rotation per sample, so cost is @f$ O(N \cdot M) @f$ with
 *       @f$ N @f$ samples and @f$ M @f$ modes, with two multiplications
 *       and one addition per (sample, mode). For tens of thousands of
 *       samples and a few hundred modes this is well below 1 ms; for
 *       larger banks see @cite bonneel_2008_fast_modal.
 */
std::vector<float> synthesize_resonator_bank(
    std::span<const ResonatorMode> modes,
    double sample_rate_hz,
    double duration_s);

}  // namespace chladni::synth
