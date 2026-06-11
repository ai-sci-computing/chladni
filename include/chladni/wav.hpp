#pragma once

/**
 * @file wav.hpp
 * @brief Minimal mono 16-bit PCM WAV file writer.
 *
 * The writer is deliberately tiny — it exists so that the synthesiser
 * can dump a buffer to disk for offline listening and validation, with
 * zero external dependencies. For real-time audio playback the strike
 * pipeline targets miniaudio directly.
 *
 * The file layout produced is a canonical 44-byte RIFF/WAVE header
 * followed by little-endian @f$s_{16}@f$ PCM samples. Endianness is
 * always written little-endian regardless of host byte order.
 */

#include <cstdint>
#include <filesystem>
#include <span>

namespace chladni::wav {

/**
 * @brief Write a mono 16-bit signed-PCM WAV file.
 *
 * Float samples are clipped to @f$[-1, 1]@f$ and quantised to int16 with
 * the convention @f$ s_{16} = \mathrm{round}(\mathrm{clip}(s_{f}) \cdot
 * 32767) @f$. The peak positive sample maps to 32767 and the peak
 * negative sample maps to -32767 (symmetric, leaving -32768 unused).
 *
 * @param path            Destination path. Parent directory must exist;
 *                        any existing file at the path is truncated.
 * @param samples         Source buffer. Empty buffer is allowed and
 *                        produces a zero-sample WAV (44-byte header
 *                        with a 0-byte data chunk).
 * @param sample_rate_hz  Sample rate written into the WAV header;
 *                        must be @f$> 0@f$ and @f$\le 2^{31}-1@f$ (so the
 *                        derived 32-bit @c byte_rate field cannot overflow).
 *
 * @throws std::invalid_argument if `sample_rate_hz` is zero or large enough
 *         that `byte_rate` would overflow `uint32`.
 * @throws std::ios_base::failure on I/O failure.
 *
 * @note 16-bit was chosen for compatibility with every audio player on
 *       the planet. 32-bit float WAV is in the spec but is less
 *       reliably consumed; we keep the writer simple and use 16-bit
 *       throughout.
 */
void write_mono16(
    const std::filesystem::path& path,
    std::span<const float> samples,
    std::uint32_t sample_rate_hz);

}  // namespace chladni::wav
