/**
 * @file test_synth.cpp
 * @brief Unit tests for the resonator-bank synthesiser.
 *
 * Pins
 * @ref chladni::synth::synthesize_resonator_bank
 * to the analytical envelope of a single damped sinusoid.
 *
 * For one resonator with @f$ \omega = 2\pi \cdot 100 @f$ rad/s,
 * @f$ d = 10 @f$ s^-1, @f$ A = 0.5 @f$, @f$ \varphi = 0 @f$,
 * sampled at 44.1 kHz over 1 s, the analytical reference values are:
 *
 *   - @f$ y(0) = 0 @f$ exactly (sin 0).
 *   - The first peak occurs at @f$ t = T/4 = 2.5 @f$ ms, where
 *     sin(omega t) = 1 and the envelope has decayed by exp(-10*0.0025)
 *     = 0.9753, giving |y| = 0.5 * 0.9753 ≈ 0.4876.
 *   - At @f$ t = 0.5 @f$ s the envelope has decayed to exp(-5) ≈ 0.00674,
 *     bounding |y| < 0.0034.
 */

#include <chladni/synth.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

using chladni::synth::ResonatorMode;
using chladni::synth::synthesize_resonator_bank;

namespace {

constexpr double kSampleRate = 44100.0;
constexpr double kDuration   = 1.0;

}  // namespace

TEST_CASE("Single-mode resonator: zero start, peak amplitude, decay envelope",
          "[synth][resonator]")
{
    const ResonatorMode mode{
        .angular_frequency = 2.0 * std::numbers::pi_v<double> * 100.0,  // 100 Hz
        .damping_rate      = 10.0,                                       // 1/s
        .amplitude         = 0.5,
        .phase             = 0.0,
    };

    const std::array<ResonatorMode, 1> bank{mode};
    const auto y = synthesize_resonator_bank(
        std::span<const ResonatorMode>{bank}, kSampleRate, kDuration);

    SECTION("buffer length matches sample_rate * duration") {
        REQUIRE(y.size() ==
                static_cast<std::size_t>(std::llround(kSampleRate * kDuration)));
    }

    SECTION("y[0] is zero (sin 0 with phi = 0)") {
        REQUIRE(std::abs(y.front()) < 1e-6f);
    }

    SECTION("first peak in first quarter-period is approximately A * env") {
        // Quarter period of 100 Hz = 2.5 ms = ~110 samples at 44.1 kHz.
        const std::size_t quarter_samples =
            static_cast<std::size_t>(std::llround(0.0025 * kSampleRate));
        REQUIRE(quarter_samples < y.size());

        const auto it = std::max_element(
            y.begin(), y.begin() + static_cast<std::ptrdiff_t>(quarter_samples + 1),
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        const double peak = std::abs(static_cast<double>(*it));

        // Reference 0.4876, allow 1% relative tolerance.
        const double expected = 0.5 * std::exp(-10.0 * 0.0025);
        const double rel_err = std::abs(peak - expected) / expected;
        INFO("peak=" << peak << " expected=" << expected << " rel_err=" << rel_err);
        REQUIRE(rel_err < 1e-2);
    }

    SECTION("late samples decay below envelope bound") {
        // At t = 0.5 s, envelope = 0.5 * exp(-5) ≈ 0.0034.
        const std::size_t late_idx =
            static_cast<std::size_t>(std::llround(0.5 * kSampleRate));
        REQUIRE(late_idx < y.size());
        REQUIRE(std::abs(static_cast<double>(y[late_idx])) < 0.0035);
    }
}

TEST_CASE("Empty bank synthesises silence", "[synth][resonator]")
{
    const auto y = synthesize_resonator_bank(
        std::span<const ResonatorMode>{}, kSampleRate, 0.01);

    REQUIRE(y.size() ==
            static_cast<std::size_t>(std::llround(kSampleRate * 0.01)));
    for (float s : y) {
        REQUIRE(s == 0.0f);
    }
}

TEST_CASE("Heavily-damped mode: decay cutoff zeros the inaudible tail exactly "
          "and leaves pre-cutoff samples intact (review4 F8)",
          "[synth][resonator]")
{
    // A heavily damped mode whose envelope slides below the 1e-30 amplitude
    // floor long before the 1 s buffer ends. The throughput optimisation
    // (commit c71afdd) stops the recurrence there; pin that it (a) emits
    // bit-exact 0.0f past the cutoff and (b) does not truncate any sample
    // while it is still audible. The existing decay test only bounds
    // magnitude, so an over-eager-truncation regression would slip past it.
    constexpr double d = 200.0;  // 1/s — rings for tens of ms
    const ResonatorMode mode{
        .angular_frequency = 2.0 * std::numbers::pi_v<double> * 1000.0,
        .damping_rate      = d,
        .amplitude         = 1.0,
        .phase             = 0.0,
    };
    const std::array<ResonatorMode, 1> bank{mode};
    const auto y = synthesize_resonator_bank(
        std::span<const ResonatorMode>{bank}, kSampleRate, kDuration);

    const double dt = 1.0 / kSampleRate;
    // Sample index where |A|*exp(-d*i*dt) crosses the 1e-30 floor; the loop
    // runs for i in [0, ceil(i_cut)), so samples at i >= ceil(i_cut) are 0.
    const double i_cut = std::log(1.0 / 1.0e-30) / (d * dt);
    const auto   cut   = static_cast<std::size_t>(std::ceil(i_cut));
    REQUIRE(cut < y.size());  // the optimisation actually engages

    SECTION("envelope at the cutoff is far below the 24-bit audio floor") {
        const double env_at_cut = std::exp(-d * static_cast<double>(cut) * dt);
        // 2^-24 ≈ 6e-8; the floor sits many orders of magnitude below it,
        // so nothing audible is dropped.
        REQUIRE(env_at_cut < 1.0e-20);
    }

    SECTION("tail past the cutoff is bit-exactly zero") {
        for (std::size_t i = cut; i < y.size(); ++i) {
            REQUIRE(y[i] == 0.0f);
        }
    }

    SECTION("an audible early sample matches the analytic envelope") {
        // t = 5 ms: envelope exp(-1) ≈ 0.368, well above the noise floor.
        const std::size_t i = static_cast<std::size_t>(
            std::llround(0.005 * kSampleRate));
        const double t   = static_cast<double>(i) * dt;
        const double ref = std::exp(-d * t)
                           * std::sin(mode.angular_frequency * t);
        INFO("i=" << i << " y=" << y[i] << " ref=" << ref);
        REQUIRE(std::abs(static_cast<double>(y[i]) - ref) < 1e-6);
    }
}

TEST_CASE("Invalid arguments throw", "[synth][resonator][validation]")
{
    const std::array<ResonatorMode, 0> empty{};
    const std::span<const ResonatorMode> empty_span{empty};

    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, 0.0, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, 44100.0, -1.0),
                      std::invalid_argument);

    // Non-finite scalars: `x <= 0.0` is false for NaN, so the weaker guard let
    // them through and dt = 1/NaN poisoned the whole buffer (review8 J2).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, nan, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, inf, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, 44100.0, nan),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, 44100.0, inf),
                      std::invalid_argument);
    // Absurd-but-finite rate×duration must be a clean domain error, not a
    // bad_alloc (review8 J6 — the wav writer caps its buffer the same way).
    REQUIRE_THROWS_AS(synthesize_resonator_bank(empty_span, 1.0e9, 1.0e6),
                      std::invalid_argument);
}

TEST_CASE("Per-mode parameters are validated (review5 G8)",
          "[synth][resonator][validation]")
{
    // The header documents damping_rate >= 0; a negative decay constant
    // makes exp(-d*dt) > 1 and the envelope grows without bound, silently
    // blowing past [-1, 1]. A non-finite omega/amplitude/phase poisons
    // the entire output buffer. Both must throw rather than corrupt audio.
    const auto run_one = [](ResonatorMode m) {
        const std::array<ResonatorMode, 1> bank{m};
        return synthesize_resonator_bank(std::span<const ResonatorMode>{bank},
                                         44100.0, 0.1);
    };

    SECTION("negative damping_rate throws")
    {
        REQUIRE_THROWS_AS(run_one({1000.0, -1.0, 1.0, 0.0}),
                          std::invalid_argument);
    }
    SECTION("NaN damping_rate throws")
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(run_one({1000.0, nan, 1.0, 0.0}),
                          std::invalid_argument);
    }
    SECTION("non-finite amplitude throws")
    {
        const double inf = std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(run_one({1000.0, 1.0, inf, 0.0}),
                          std::invalid_argument);
    }
    SECTION("non-finite omega throws")
    {
        const double inf = std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(run_one({inf, 1.0, 1.0, 0.0}),
                          std::invalid_argument);
    }
    SECTION("non-finite phase throws")
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(run_one({1000.0, 1.0, 1.0, nan}),
                          std::invalid_argument);
    }
    SECTION("valid zero damping is accepted")
    {
        REQUIRE_NOTHROW(run_one({1000.0, 0.0, 1.0, 0.0}));
    }
}
