/**
 * @file test_wav.cpp
 * @brief Unit tests for the mono 16-bit PCM WAV writer.
 *
 * Verifies bit-level RIFF/WAVE conformance and quantisation round-trip:
 *   - first four bytes = "RIFF"
 *   - bytes 8..11 = "WAVE"
 *   - fmt chunk reports format=PCM, channels=1, bits_per_sample=16
 *   - file size matches header + data
 *   - quantised samples round-trip back to the source within ~1/32768
 */

#include <chladni/wav.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kSampleRate = 44100;

/// Read an entire binary file into a vector of bytes.
std::vector<std::uint8_t> read_file(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    REQUIRE(f.good());
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(size));
    REQUIRE(f.good());
    return buf;
}

/// Decode a little-endian uint16 starting at byte offset @p off.
std::uint16_t le_u16(const std::vector<std::uint8_t>& b, std::size_t off)
{
    return static_cast<std::uint16_t>(b[off] | (b[off + 1] << 8));
}

/// Decode a little-endian uint32 starting at byte offset @p off.
std::uint32_t le_u32(const std::vector<std::uint8_t>& b, std::size_t off)
{
    return static_cast<std::uint32_t>(b[off])
         | (static_cast<std::uint32_t>(b[off + 1]) << 8)
         | (static_cast<std::uint32_t>(b[off + 2]) << 16)
         | (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

/// Decode a little-endian int16 starting at byte offset @p off.
std::int16_t le_i16(const std::vector<std::uint8_t>& b, std::size_t off)
{
    return static_cast<std::int16_t>(le_u16(b, off));
}

}  // namespace

TEST_CASE("WAV writer: a 100-sample sine wave produces a valid RIFF file",
          "[wav][round_trip]")
{
    constexpr std::size_t N = 100;
    std::vector<float> samples(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSampleRate);
        samples[i] = static_cast<float>(
            0.5 * std::sin(2.0 * std::numbers::pi_v<double> * 440.0 * t));
    }

    const fs::path tmp = fs::temp_directory_path() / "chladni_test_sine.wav";
    fs::remove(tmp);  // ensure clean slate

    chladni::wav::write_mono16(tmp,
                               std::span<const float>{samples},
                               kSampleRate);

    REQUIRE(fs::exists(tmp));
    const auto bytes = read_file(tmp);

    SECTION("file is 44-byte header + 2 bytes/sample") {
        REQUIRE(bytes.size() == 44 + 2 * N);
    }

    SECTION("RIFF / WAVE magic") {
        REQUIRE(std::memcmp(bytes.data(),     "RIFF", 4) == 0);
        REQUIRE(std::memcmp(bytes.data() + 8, "WAVE", 4) == 0);
    }

    SECTION("RIFF chunk size = file_size - 8") {
        REQUIRE(le_u32(bytes, 4) == bytes.size() - 8);
    }

    SECTION("fmt subchunk advertises mono 16-bit PCM at the requested rate") {
        REQUIRE(std::memcmp(bytes.data() + 12, "fmt ", 4) == 0);
        REQUIRE(le_u32(bytes, 16) == 16);              // PCM fmt size
        REQUIRE(le_u16(bytes, 20) == 1);               // audio format = PCM
        REQUIRE(le_u16(bytes, 22) == 1);               // channels
        REQUIRE(le_u32(bytes, 24) == kSampleRate);     // sample rate
        REQUIRE(le_u32(bytes, 28) == kSampleRate * 2); // byte rate
        REQUIRE(le_u16(bytes, 32) == 2);               // block align
        REQUIRE(le_u16(bytes, 34) == 16);              // bits per sample
    }

    SECTION("data subchunk size matches sample count") {
        REQUIRE(std::memcmp(bytes.data() + 36, "data", 4) == 0);
        REQUIRE(le_u32(bytes, 40) == 2 * N);
    }

    SECTION("samples round-trip through int16 quantisation") {
        for (std::size_t i = 0; i < N; ++i) {
            const auto offset = 44 + 2 * i;
            const auto q = le_i16(bytes, offset);
            const auto reconstructed = static_cast<float>(q) / 32767.0f;
            // Quantisation step is 1/32767. Allow 2 LSB of slack for
            // round-to-nearest plus float -> double -> float.
            REQUIRE(std::abs(reconstructed - samples[i]) < (2.0f / 32767.0f));
        }
    }

    fs::remove(tmp);
}

TEST_CASE("WAV writer: empty buffer produces a header-only file",
          "[wav][edge]")
{
    const fs::path tmp = fs::temp_directory_path() / "chladni_test_empty.wav";
    fs::remove(tmp);

    chladni::wav::write_mono16(tmp,
                               std::span<const float>{},
                               kSampleRate);

    REQUIRE(fs::exists(tmp));
    const auto bytes = read_file(tmp);
    REQUIRE(bytes.size() == 44);
    REQUIRE(std::memcmp(bytes.data(), "RIFF", 4) == 0);
    REQUIRE(le_u32(bytes, 40) == 0);  // data subchunk is empty

    fs::remove(tmp);
}

TEST_CASE("WAV writer: non-finite samples quantise to silence, not a click "
          "(review6 H4)",
          "[wav][edge]")
{
    // std::clamp returns NaN unchanged (both comparisons false), so without
    // an isfinite guard std::lround(NaN) ran — implementation-defined,
    // typically a full-scale int16 click. Non-finite input must map to 0.
    const fs::path tmp = fs::temp_directory_path() / "chladni_test_nonfinite.wav";
    fs::remove(tmp);

    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 4> samples{nan, inf, -inf, 0.5f};

    chladni::wav::write_mono16(tmp,
                               std::span<const float>{samples},
                               kSampleRate);

    const auto bytes = read_file(tmp);
    REQUIRE(bytes.size() == 44 + 2 * samples.size());
    REQUIRE(le_i16(bytes, 44) == 0);  // NaN  -> silence
    REQUIRE(le_i16(bytes, 46) == 0);  // +inf -> silence
    REQUIRE(le_i16(bytes, 48) == 0);  // -inf -> silence
    // The finite sample still quantises normally.
    REQUIRE(le_i16(bytes, 50) == static_cast<std::int16_t>(
                                     std::lround(0.5 * 32767.0)));

    fs::remove(tmp);
}

TEST_CASE("WAV writer: zero sample rate is rejected",
          "[wav][validation]")
{
    const fs::path tmp = fs::temp_directory_path() / "chladni_test_invalid.wav";
    fs::remove(tmp);

    const std::array<float, 0> empty{};
    REQUIRE_THROWS_AS(
        chladni::wav::write_mono16(tmp, std::span<const float>{empty}, 0u),
        std::invalid_argument);
}
