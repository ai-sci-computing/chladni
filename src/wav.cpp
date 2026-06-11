/**
 * @file wav.cpp
 * @brief Implementation of the mono 16-bit PCM WAV writer.
 *
 * The output is a canonical 44-byte RIFF / WAVE header followed by
 * little-endian int16 samples:
 *
 * @verbatim
 *   offset  size  field
 *   ----------------------
 *      0     4    "RIFF"
 *      4     4    riff_size = 36 + data_bytes
 *      8     4    "WAVE"
 *     12     4    "fmt "
 *     16     4    16                    // PCM fmt chunk size
 *     20     2    1                     // audio format = PCM
 *     22     2    1                     // num_channels = mono
 *     24     4    sample_rate_hz
 *     28     4    sample_rate_hz * 2    // byte rate
 *     32     2    2                     // block align
 *     34     2    16                    // bits per sample
 *     36     4    "data"
 *     40     4    data_bytes = num_samples * 2
 *     44   N*2    samples (le int16)
 * @endverbatim
 *
 * Endianness is always written little-endian regardless of host order.
 */

#include <chladni/wav.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

namespace chladni::wav {

namespace {

/// Write an unsigned 16-bit value little-endian to @p os.
void write_le_u16(std::ostream& os, std::uint16_t v)
{
    const char b[2] = {
        static_cast<char>(v & 0xffu),
        static_cast<char>((v >> 8) & 0xffu),
    };
    os.write(b, 2);
}

/// Write an unsigned 32-bit value little-endian to @p os.
void write_le_u32(std::ostream& os, std::uint32_t v)
{
    const char b[4] = {
        static_cast<char>(v & 0xffu),
        static_cast<char>((v >> 8) & 0xffu),
        static_cast<char>((v >> 16) & 0xffu),
        static_cast<char>((v >> 24) & 0xffu),
    };
    os.write(b, 4);
}

/// Write a fixed 4-character FOURCC tag (no NUL) to @p os.
void write_fourcc(std::ostream& os, const char (&tag)[5])
{
    os.write(tag, 4);
}

}  // namespace

void write_mono16(
    const std::filesystem::path& path,
    std::span<const float> samples,
    std::uint32_t sample_rate_hz)
{
    if (sample_rate_hz == 0) {
        throw std::invalid_argument(
            "write_mono16: sample_rate_hz must be > 0");
    }
    // byte_rate = sample_rate_hz * block_align (2) is written into a 32-bit
    // WAV header field; reject rates that would wrap it. The contract's data
    // chunk is promoted to uint64 before its overflow check (below) but
    // byte_rate was not — this closes that one-field-promoted-its-sibling-not
    // asymmetry (review7 I9).
    if (sample_rate_hz > 0xffffffffu / 2u) {
        throw std::invalid_argument(
            "write_mono16: sample_rate_hz too large — byte_rate would "
            "overflow the 32-bit WAV header field");
    }

    const auto data_bytes_u64 = static_cast<std::uint64_t>(samples.size()) * 2;
    if (data_bytes_u64 > static_cast<std::uint64_t>(0xffffffffu) - 36u) {
        // 4 GiB - 36 (the 36 bytes of header that share the RIFF size field)
        throw std::invalid_argument(
            "write_mono16: sample buffer exceeds 4 GiB WAV limit");
    }
    const auto data_bytes  = static_cast<std::uint32_t>(data_bytes_u64);
    const auto riff_size   = 36u + data_bytes;
    const auto byte_rate   = sample_rate_hz * 2u;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::ios_base::failure(
            "write_mono16: failed to open " + path.string());
    }
    f.exceptions(std::ios::failbit | std::ios::badbit);

    // RIFF chunk header.
    write_fourcc(f, "RIFF");
    write_le_u32(f, riff_size);
    write_fourcc(f, "WAVE");

    // fmt subchunk (PCM).
    write_fourcc(f, "fmt ");
    write_le_u32(f, 16u);                  // PCM fmt size
    write_le_u16(f, 1u);                   // audio format = PCM
    write_le_u16(f, 1u);                   // channels = mono
    write_le_u32(f, sample_rate_hz);
    write_le_u32(f, byte_rate);
    write_le_u16(f, 2u);                   // block align
    write_le_u16(f, 16u);                  // bits per sample

    // data subchunk header.
    write_fourcc(f, "data");
    write_le_u32(f, data_bytes);

    // Sample payload: clip to [-1, 1], symmetric quantisation through 32767.
    // A non-finite sample slips through std::clamp unchanged (both s<lo and
    // hi<s are false for NaN), so std::lround(NaN) would run — implementation-
    // defined, typically a full-scale int16 click. Map non-finite to silence
    // to honour the documented [-1, 1] clip contract on this public API.
    for (float s : samples) {
        const float c = std::isfinite(s) ? std::clamp(s, -1.0f, 1.0f) : 0.0f;
        const auto q = static_cast<std::int16_t>(
            std::lround(static_cast<double>(c) * 32767.0));
        write_le_u16(f, static_cast<std::uint16_t>(q));
    }

    // Close explicitly so a flush failure (e.g. a full disk on the final
    // buffered write) is reported. The ofstream destructor would flush
    // silently and swallow any error, leaving a truncated file undetected.
    // exceptions() is still armed, so a failed flush/close throws here.
    f.close();
}

}  // namespace chladni::wav
