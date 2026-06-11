/**
 * @file main.cpp
 * @brief strike_gui — interactive shell-strike viewer with realtime audio.
 *
 * Phase 3 / F1+F2+F3+F4. End-to-end flow:
 *
 *   1. Load a triangle mesh (default cylinder.obj).
 *   2. Compute its lowest 30 vibration modes via @ref chladni::shell.
 *   3. Render the mesh in Polyscope; show an ImGui panel with material
 *      sliders (E, nu, rho, h), a hammer-type dropdown, and a
 *      "Recompute modes" button.
 *   4. On left-mouse click on the mesh, project an inward-normal
 *      impulse at the clicked vertex onto every mode and overwrite the
 *      audio state so the resonator bank rings out the strike.
 *   5. miniaudio runs a CoreAudio (macOS) playback callback at 44.1 kHz
 *      that walks each mode forward by one complex rotation per sample
 *      (the same trick used in @ref chladni::synth::synthesize_resonator_bank,
 *      adapted to streaming).
 *
 * Concurrency: a single std::mutex guards the audio mode bank. The audio
 * thread acquires it with try_lock to avoid blocking on the main thread;
 * if the lock is busy (a recompute / strike is in progress) the buffer
 * for that callback is filled with silence, which is a 5–20 ms gap and
 * inaudible in practice.
 */

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <filesystem>

#include <chladni/material.hpp>
#include <chladni/mesh.hpp>
#include <chladni/shell.hpp>
#include <chladni/shell/assembler.hpp>
#include <chladni/shell/lme.hpp>
#include <chladni/shell/loop.hpp>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/surface_scalar_quantity.h>
#include <polyscope/types.h>
#include <polyscope/view.h>

#include <imgui.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Audio state — touched by the audio thread.
// ---------------------------------------------------------------------------

struct AudioMode {
    double               angular_frequency = 0.0;
    double               damping_rate      = 0.0;
    std::complex<double> step              = {1.0, 0.0};  // exp(-d*dt) * exp(i*omega*dt)
    std::complex<double> z                 = {0.0, 0.0};  // current state
    // A mode whose per-sample phase advance ω·dt ≥ π is above the audio
    // Nyquist frequency: stepping it aliases to a spurious lower pitch.
    // Such modes are excluded from the audio sum (the visualiser keeps them).
    // Recomputed wherever `step` is — dt tracks the slowdown slider.
    bool                 audible           = true;
};

struct AudioState {
    std::vector<AudioMode> modes;
    std::mutex             mutex;
    double                 sample_rate = 44100.0;
};

// ---------------------------------------------------------------------------
// Hammer model.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Material presets.
//
// Reference values for common shell-instrument materials (E in GPa, nu
// dimensionless, rho in kg/m^3). These are picks from machinery /
// acoustics handbooks; fine to tweak. The dropdown sets the three
// material sliders; touching any slider afterwards flips the dropdown
// to "Custom".
// ---------------------------------------------------------------------------

constexpr const char* kMaterialPresetNames[] = {
    "Steel", "Aluminum", "Brass", "Copper", "Titanium",
    "Glass", "Maple (wood)", "Acrylic", "Custom"};
constexpr int kMaterialPresetCustom = 8;

/// A material preset bundles four values: the three elastic
/// parameters (E, nu, rho) and the loss factor (eta). The loss factor
/// is the natural human-friendly damping dial: dimensionless, common
/// across machinery / acoustics handbooks, and approximately constant
/// across audible frequencies for typical structural materials.
///
/// Rayleigh @c (alpha, beta) is the per-mode parameterisation the
/// audio bank wants: per-mode decay rate is
/// @f$d_i = (\alpha + \beta\,\omega_i^2)/2 \approx (\eta/2)\,\omega_i@f$
/// and the approximation holds **exactly** at two anchor frequencies
/// when
/// @f$\,\beta = \eta / (\omega_a + \omega_b)\,@f$ and
/// @f$\,\alpha = \eta\,\omega_a\,\omega_b / (\omega_a + \omega_b)\,@f$.
/// We anchor at 200 Hz and 5 kHz: modes in that band get damping
/// close to the true constant loss factor, modes outside drift slightly.
///
/// Loss factors taken from Cremer/Heckl, "Structure-Borne Sound":
///     steel/aluminum  eta ~ 1-2e-4   (very low; rings for seconds)
///     brass/copper    eta ~ 1-2e-3
///     glass / titanium eta ~ 0.5-2e-3
///     wood (maple)    eta ~ 1e-2     (100x higher than steel)
///     acrylic         eta ~ 4e-2     (rings for ~50 ms before dead)
struct MaterialPreset {
    float E_GPa;
    float nu;
    float rho;
    float eta;    ///< loss factor (dimensionless)
};

constexpr MaterialPreset kMaterialPresets[] = {
    // E       nu     rho       eta
    {200.0F, 0.30F, 7850.0F,  2.0e-4F},  // Steel
    { 69.0F, 0.33F, 2700.0F,  1.0e-4F},  // Aluminum
    {105.0F, 0.34F, 8500.0F,  1.0e-3F},  // Brass
    {117.0F, 0.34F, 8960.0F,  2.0e-3F},  // Copper
    {116.0F, 0.32F, 4500.0F,  6.0e-4F},  // Titanium
    { 70.0F, 0.22F, 2500.0F,  2.0e-3F},  // Glass
    { 12.0F, 0.30F,  720.0F,  1.0e-2F},  // Maple wood
    {  3.2F, 0.37F, 1180.0F,  4.0e-2F},  // Acrylic
};

/// Anchor angular frequencies for the two-point Rayleigh fit.
/// @f$\omega_a@f$ = 200 Hz, @f$\omega_b@f$ = 5 kHz.
constexpr double kEtaAnchorOmegaA = 2.0 * std::numbers::pi_v<double> *  200.0;
constexpr double kEtaAnchorOmegaB = 2.0 * std::numbers::pi_v<double> * 5000.0;

/// @brief Convert a loss factor @p eta into Rayleigh (alpha, beta).
/// See @ref MaterialPreset Doxygen for the derivation.
inline constexpr std::pair<double, double> rayleigh_from_eta(double eta)
{
    constexpr double sum_w  = kEtaAnchorOmegaA + kEtaAnchorOmegaB;
    constexpr double prod_w = kEtaAnchorOmegaA * kEtaAnchorOmegaB;
    return {eta * prod_w / sum_w, eta / sum_w};
}

/// @brief Recover the loss factor from a hand-set @p alpha — the exact inverse
/// of @ref rayleigh_from_eta along the mass-proportional axis.
///
/// Rayleigh damping is genuinely two-parameter, so a freely dragged
/// (@c alpha, @c beta) pair need not lie on the one-parameter @c eta fit line.
/// This recovers the @c eta whose two-anchor fit reproduces the current
/// @c alpha, so the master @c eta dial stays consistent with the axis the user
/// just touched (@c beta is left untouched, preserving the second degree of
/// freedom). Inverse of @c alpha = eta * prod_w / sum_w.
inline constexpr double eta_from_alpha(double alpha)
{
    constexpr double sum_w  = kEtaAnchorOmegaA + kEtaAnchorOmegaB;
    constexpr double prod_w = kEtaAnchorOmegaA * kEtaAnchorOmegaB;
    return alpha * sum_w / prod_w;
}

/// @brief Recover the loss factor from a hand-set @p beta — the exact inverse
/// of @ref rayleigh_from_eta along the stiffness-proportional axis.
/// Inverse of @c beta = eta / sum_w. See @ref eta_from_alpha for the rationale.
inline constexpr double eta_from_beta(double beta)
{
    constexpr double sum_w = kEtaAnchorOmegaA + kEtaAnchorOmegaB;
    return beta * sum_w;
}

// Per-axis round-trip: eta -> (alpha, beta) -> eta is exact on the fit line.
// Pins eta_from_alpha / eta_from_beta as the genuine inverses of
// rayleigh_from_eta at compile time (these app-local helpers have no
// test-binary linkage, so the contract is enforced here instead).
namespace damping_roundtrip_check {
constexpr double                   kProbeEta = 1.0e-2;
constexpr std::pair<double, double> kProbeAB  = rayleigh_from_eta(kProbeEta);
constexpr double                   kErrAlpha = eta_from_alpha(kProbeAB.first)  - kProbeEta;
constexpr double                   kErrBeta  = eta_from_beta(kProbeAB.second) - kProbeEta;
static_assert(kErrAlpha > -1e-16 && kErrAlpha < 1e-16,
              "eta_from_alpha must invert rayleigh_from_eta's alpha component");
static_assert(kErrBeta > -1e-16 && kErrBeta < 1e-16,
              "eta_from_beta must invert rayleigh_from_eta's beta component");
}  // namespace damping_roundtrip_check

// Two parameters describe a hammer strike:
//
//   - "Force" (relative): a unit-free multiplier on the impulse
//     magnitude. The hard-mallet reference is 1.0.
//   - "Cutoff" (Hz): finite contact time tau in [0, ~30 ms] gives a
//     spectral roll-off A(omega) = 1/sqrt(1 + (omega/omega_c)^2) with
//     omega_c ≈ 1/tau, attenuating modes well above the knee. A
//     30 kHz cutoff is effectively "flat" for any audible mode.
//
// The dropdown only sets the two sliders to preset values; the user
// can drift either slider freely afterwards (the dropdown then becomes
// "Custom" until a preset is re-selected).

constexpr const char* kHammerPresetNames[] = {
    "Hard mallet", "Drumstick", "Soft mallet", "Hand", "Custom"};
constexpr int kHammerPresetCustom = 4;

struct HammerPreset { float force; float cutoff_hz; };

constexpr HammerPreset kHammerPresets[] = {
    {1.00F, 30000.0F},  // Hard mallet — essentially flat
    {0.70F,  8000.0F},  // Drumstick
    {0.40F,  2000.0F},  // Soft mallet
    {0.25F,   500.0F},  // Hand
};

// ---------------------------------------------------------------------------
// Compute-parameter snapshot. Captured at kick_off_recompute time so
// the eventual apply_modes_result has the actual inputs the worker
// used (independent of slider edits that happen mid-compute) and so
// the diagnostic log doesn't lie about which h / E / n_passes
// produced the spectrum it's printing alongside.
// ---------------------------------------------------------------------------

/// Which ShellAssembler implementation the modal solver should use.
/// Selection is driven by which tab is active in the formulation TabBar;
/// the worker captures this into @ref ComputeParams at kick-off so live
/// edits cannot leak into an in-flight compute.
enum class AssemblerKind {
    Loop,     ///< Cirak-Ortiz-Schroder 2000 (C¹ subdivision FEM)
    LME,      ///< Arroyo-Ortiz 2006 meshfree (flat plates only)
    Legacy    ///< CST + Wardetzky-IBM dihedral hinge (fallback)
};

struct ComputeParams {
    float  E_GPa;
    float  nu;
    float  rho;
    float  h_mm;
    int    n_modes_request;
    AssemblerKind assembler;
    chladni::shell::LoopAssembler::Params loop;  ///< snapshot of Loop knobs
    chladni::shell::LMEAssembler::Params  lme;   ///< snapshot of LME knobs
    std::string assembly_label;
    Eigen::Index expected_dofs;  // 3 * mesh.V.rows() at kick_off
};

// ---------------------------------------------------------------------------
// App state — touched by the main (UI) thread.
// ---------------------------------------------------------------------------

struct AppState {
    // ---- Mesh source. The directory is scanned once at startup so the
    // Mesh dropdown lists every OBJ shipped with the project. The user
    // can swap meshes mid-session; load_mesh_from_path rebuilds normals,
    // extrusion, Polyscope structure, and modes from scratch.
    fs::path                  mesh_dir;
    /// Display labels for the dropdown, e.g. "torus.obj (44 KB)". Built
    /// from the same scan as @c mesh_choice_filenames so indices stay in
    /// lockstep.
    std::vector<std::string>  mesh_choices;
    /// Parallel @c const @c char* view of @c mesh_choices for
    /// @c ImGui::Combo. Rebuilt on every scan.
    std::vector<const char*>  mesh_choice_cstrs;
    /// Bare filenames ("torus.obj"), parallel to @c mesh_choices. Used
    /// to build the on-disk path when the user picks an entry.
    std::vector<std::string>  mesh_choice_filenames;
    int                       mesh_choice_idx = 0;
    std::string               current_mesh_name;  ///< what is currently loaded

    chladni::mesh::TriMesh mesh;             ///< current rest configuration (mid-surface)
    Eigen::MatrixXd        V_rest;           ///< saved rest vertices for viz (mid-surface)
    polyscope::SurfaceMesh* ps_mesh = nullptr;

    // Extruded mesh used for rendering only. Each mid-surface vertex i
    // gets at least two extruded copies (outer at +h/2, inner at -h/2);
    // boundary vertices additionally get duplicates that the side-wall
    // ribbon faces use exclusively, so smooth shading produces a crisp
    // 90 degree crease at the rim without averaging side-wall normals
    // into the outer-surface vertex normals.
    Eigen::MatrixXd V_rest_ext;       ///< rest extruded vertices, may exceed 2 n_mid
    Eigen::MatrixXi F_ext;            ///< extruded face indices
    Eigen::Index    n_mid_vertices = 0;  ///< number of coarse mid-surface vertices (FEM)
    /// Length-V_rest_ext.rows() map: extruded row -> *viz* (subdivided)
    /// mid-surface vertex it follows. With viz_subdivision_passes=0,
    /// viz vertices coincide with coarse vertices; with passes > 0, the
    /// viz mesh is the coarse mesh Loop-subdivided that many times.
    std::vector<Eigen::Index> ext_to_mid;

    Eigen::MatrixXd vertex_normals;  ///< @c n_mid x 3 unit outward normals on the COARSE mesh (used by apply_strike, which maps clicks to coarse vertices).

    // ---- Visualization-only mesh refinement (independent of FEM).
    // The eigenproblem is solved on the coarse mesh (app.mesh); the
    // rendered surface and nodal-line overlay are computed on a
    // Loop-subdivided copy. This makes Polyscope's piecewise-linear
    // CONTOUR shader interpolate across many small triangles instead
    // of a few large coarse ones, dramatically smoothing the nodal
    // lines of high-m modes. The subdivision matrix S_viz maps the
    // 3 n_coarse-vector modal displacement to the 3 n_viz-vector that
    // the renderer consumes; with passes=0, mesh_viz == app.mesh and
    // S_viz is empty (treated as identity in update_viz_mesh).
    int                          viz_subdivision_passes = 1;  ///< 0, 1, or 2
    chladni::mesh::TriMesh       mesh_viz;
    Eigen::SparseMatrix<double>  S_viz;
    Eigen::Index                 n_viz_vertices = 0;
    Eigen::MatrixXd              vertex_normals_viz;

    // Material (UI uses floats for ImGui sliders, converted to double when used).
    int   material_preset_idx = 0;                       ///< Steel
    float E_GPa = kMaterialPresets[0].E_GPa;
    float nu    = kMaterialPresets[0].nu;
    float rho   = kMaterialPresets[0].rho;
    float h_mm  = 1.0F;

    // Damping. The user-facing dial is the dimensionless loss factor
    // eta; alpha and beta are the per-mode-step parameters the audio
    // bank actually consumes. Eta drives alpha/beta on every change;
    // alpha/beta can also be edited directly (advanced override) which
    // leaves eta as a "stale" record of the last eta-driven setting.
    float eta   = kMaterialPresets[0].eta;
    float alpha;
    float beta;

    chladni::shell::ShellModes modes;
    bool                       modes_valid = false;
    std::string                last_status;
    int                        n_modes_request = 30;  ///< how many modes the eigensolve produces

    // Stiffness/mass assembly path. The Loop subdivision (Cirak-Ortiz 2000)
    // tab is the modern C¹ formulation calibrated against analytic beam
    // / plate / shell references; the legacy CST + Wardetzky-IBM tab is
    // over-stiff by ~60% on flat-plate fixtures (W-series calibration
    // gap) and kept only as an emergency fall-back if Loop's boundary-
    // vertex constraints reject a mesh (e.g. valence-2 strip corners).
    //
    // `assembler` mirrors which formulation tab is currently selected.
    // `loop_params` / `lme_params` hold the per-tab knobs (Stam, n_passes,
    // mass lumping, K/M quadrature; γ) — captured into ComputeParams at
    // kick-off so live slider edits cannot leak into an in-flight worker.
    AssemblerKind                         assembler = AssemblerKind::Loop;
    chladni::shell::LoopAssembler::Params loop_params{};
    chladni::shell::LMEAssembler::Params  lme_params{};

    // Mode-gallery state. The gallery saves one screenshot per mode
    // (frozen at maximum displacement) into gallery_dir over
    // gallery_count successive frames. Set up via a UI button; the
    // user callback executes one step per frame.
    bool        gallery_active = false;
    int         gallery_index  = 0;
    int         gallery_count  = 20;
    float       gallery_amp    = 1.0e-3F;
    std::string gallery_dir;  // populated when the user clicks Save

    // Procedural-mesh resolution knobs. Each procedural geometry has
    // its own (n_azimuthal, n_radial) — or for the cylinder, (n_around,
    // n_along). load_procedural_mesh reads from these AppState fields,
    // so the user can adjust resolution from the GUI without touching
    // the source. The 2026-05-14 cylinder convergence audit showed
    // ~3000+ V is needed to push the chirality bias below the 0.1%
    // doublet-split level; the cylinder defaults below give 32x16 =
    // 528 V (a reasonable default; bump to 96x32 = 3168 V for clean
    // Chladni viz).
    int proc_disk_n_az      = 144;
    int proc_disk_n_rad     = 14;
    int proc_annulus_n_az   = 144;
    int proc_annulus_n_rad  = 16;
    float proc_annulus_b_over_a = 0.5F;
    int proc_cylinder_n_around = 32;
    int proc_cylinder_n_along  = 16;
    int proc_icosphere_subdivisions = 2;  // k=2 → 162 V / 320 F

    // Quad-splitting scheme for the structured procedural meshes
    // (disk / annulus / cylinder) — see chladni::mesh::QuadSplit for
    // the measured trade-offs. The legacy Consistent split keeps C_N
    // rotation but breaks every reflection, so the degenerate Chladni
    // doublets come out visibly skewed (the chirality the 2026-06-03
    // investigation root-caused). Default = UnionJack: full dihedral
    // symmetry (no chirality), uniform rims, and 12×/7.7× better
    // Leissa-disk accuracy at same-resolution / matched-DOF on the
    // SME path. CAVEAT: both symmetric schemes leave rim vertices of
    // valence ≠ 4, which the LOOP assembler's boundary augmentation
    // rejects — switch back to Consistent before using the Loop tab.
    // The LME tab (1st- and 2nd-order) handles all three splits
    // cleanly since the 2026-06-03 per-node-β derivative fix.
    chladni::mesh::QuadSplit proc_quad_split =
        chladni::mesh::QuadSplit::UnionJack;

    // Async modal-solve worker. The eigensolve on dense closed meshes
    // (genus-3, fertility, bunny @ k>=3) can take tens of seconds; we
    // run it on a detached worker thread so the UI stays responsive
    // and the window's close button remains live. The future carries
    // either a successful ShellModes or an exception; compute_params
    // snapshots the inputs at kick-off so the eventual status line
    // and diagnostic log reflect what the worker actually used (not
    // whatever the sliders happen to read at harvest time).
    std::future<chladni::shell::ShellModes> compute_future;
    bool                                    compute_in_flight = false;
    ComputeParams                           compute_params;

    // Hammer parameters (live sliders). The preset index is purely a
    // label: when the user picks a preset from the dropdown the two
    // sliders are set to its kHammerPresets values; when the user
    // moves either slider the index reverts to kHammerPresetCustom.
    int   hammer_preset_idx  = 0;                                ///< Hard mallet
    float hammer_force       = kHammerPresets[0].force;          ///< 0..2 reasonable
    float hammer_cutoff_hz   = kHammerPresets[0].cutoff_hz;      ///< 50..50000 Hz log

    // Master strike scale. With mass-orthonormal phi (typical magnitude
    // ~1) and modal amplitude a_k = phi.f/omega, a "1-Newton" impulse
    // gives ~1e-4 mode amplitudes — inaudible. Scaling the force by ~50
    // brings sustained sample peaks near 0.05 with the audio gain of
    // 0.3, which is comfortably audible.
    float strike_intensity = 50.0F;

    // Audio band-pass at strike time, applied to each mode's initial
    // amplitude on top of the hammer-cutoff roll-off. The slider
    // BOUNDS are clamped to the actually-computed spectrum so the
    // user never dials in a value that nothing in the bank can hear.
    // Defaults are set to those bounds on every recompute, i.e. the
    // band starts wide-open and the user can only drag it inward.
    float audio_low_cut_hz   = 20.0F;
    float audio_high_cut_hz  = 20000.0F;
    float spectrum_min_hz    = 20.0F;     ///< lowest computed frequency (Hz)
    float spectrum_max_hz    = 20000.0F;  ///< highest computed frequency (Hz)

    int strike_count = 0;

    // ---- Excitation mode: hammer-strike (click on mesh) vs single-mode
    // (slider-driven). Mutually exclusive — picking one greys out the
    // other's controls and disables its corresponding interaction (in
    // single-mode, click-to-strike is off entirely; in hammer mode, the
    // single-mode panel is greyed and slider edits don't ring anything).
    enum class ExcitationMode { Hammer = 0, SingleMode = 1 };
    ExcitationMode excitation_mode = ExcitationMode::Hammer;

    // ---- Single-mode excitation: educational demo for stationary
    // Chladni patterns. Setting z_k = (amp, 0) on a single mode (with
    // all others zeroed) drives a pure normal mode whose nodal lines
    // stand still on the surface, provided damping is small. Always
    // live while excitation_mode == SingleMode — every slider edit
    // immediately re-rings the selected mode (no explicit button).
    int   single_mode_k_one_based = 1;     ///< 1..n_modes
    float single_mode_amplitude   = 1.0F;  ///< modal coordinate magnitude
    /// When false, single-mode drives the visualisation only — the
    /// audio bank stays silent. Default off: the textbook Chladni
    /// use case is visual, not aural.
    bool  single_mode_sound       = false;

    AudioState* audio = nullptr;

    // ---- Slowdown couples both viz and audio: dilation factor by which
    // modal time runs slower than wall-clock time. slowdown=1 plays
    // sound at original pitch with real-time-fast deformation;
    // slowdown=N drops the audio by a factor N (so a 1 kHz mode sounds
    // at 1000/N Hz) AND the visible motion is N times slower.
    std::vector<std::complex<double>> viz_step;  ///< per-mode complex step at viz dt
    std::vector<std::complex<double>> viz_z;     ///< per-mode complex state
    float viz_slowdown      = 1.0F;     ///< 1 = real-time, no audio pitch shift
    float viz_amplification = 0.02F;    ///< multiplier on raw modal displacement
    bool  viz_enabled       = true;
    double viz_last_time    = 0.0;      ///< wall-clock at last frame (for dt)

    // Render-only thickness multiplier: the rendered wall extrudes by
    // h * viz_thickness_scale, leaving the physical h (which drives K)
    // untouched. Lets a 1 mm steel wall look palpable on a 100 mm shell.
    float viz_thickness_scale = 5.0F;

    // Chladni-pattern overlay: signed normal displacement on every
    // mid-surface vertex (replicated to the outer+inner extruded copies),
    // shown via a diverging colormap with isolines. Off by default;
    // the user activates it via the "Chladni overlay" checkbox.
    polyscope::SurfaceVertexScalarQuantity* ps_chladni_q = nullptr;
    bool   viz_chladni_enabled = false;
    /// Number of isolines drawn across the strike's symmetric range.
    /// Period is set to 2*max/N, which makes 0 always coincide with an
    /// isoline — so at N = 1 the only contour visible inside [-max, max]
    /// is the zero level, i.e. the actual Chladni nodal set. N > 1 adds
    /// equally-spaced bands above and below for finer level-set detail.
    float  viz_isoline_count     = 1.0F;
    float  viz_isoline_thickness = 0.3F;  ///< 0..1, contour line width
    float  viz_isoline_darkness  = 0.9F;  ///< 0..1, contour line darkness
    /// Below this fraction of the per-frame max |amplitude|, replace
    /// the rendered scalar with a value that lands between contour
    /// bands (no isoline drawn). Suppresses the fractal "noise floor"
    /// pattern that appears near the centre of high-m modes on the
    /// polar mesh (Bessel J_m(r) ≈ r^m → amplitude vanishes faster
    /// than the FEM / Loop-basis truncation noise). Default 0 = off.
    /// Typical useful values: 1e-4 to 1e-2.
    float  viz_chladni_noise_floor = 0.0F;
};

// ---------------------------------------------------------------------------
// Geometry helpers.
// ---------------------------------------------------------------------------

/// @brief Rescale a triangle mesh in place so its longest bounding-box
/// dimension equals @p target_size_m.
///
/// Every loaded mesh — cylinder.obj, torus.obj, bunny, etc. — encodes
/// its vertices in whatever unit the modeller picked (meters, decimeters,
/// raw blender units...). This helper imposes a single physical size on
/// the longest axis so the modal frequencies, audio bandwidth, and
/// visual scale stay comparable across meshes. The default of 20 cm is
/// roughly drum-shell size and keeps the lowest bending modes inside
/// the audible band on a fine triangulation.
///
/// @param mesh           Mesh to rescale; its V matrix is multiplied in
///                       place by the returned uniform scale factor.
/// @param target_size_m  Desired physical extent (meters) of the longest
///                       bounding-box axis. Default 0.20 m.
/// @return The uniform scale factor that was applied. Returns 1.0 (no
///         rescale) for an empty or degenerate mesh.
double rescale_to_target_size(chladni::mesh::TriMesh& mesh,
                              double target_size_m = 0.20)
{
    if (mesh.V.rows() == 0) return 1.0;
    const Eigen::Vector3d mn = mesh.V.colwise().minCoeff();
    const Eigen::Vector3d mx = mesh.V.colwise().maxCoeff();
    const double longest = (mx - mn).maxCoeff();
    if (longest <= 0.0) return 1.0;
    const double s = target_size_m / longest;
    mesh.V *= s;
    return s;
}

Eigen::MatrixXd compute_vertex_normals(const chladni::mesh::TriMesh& mesh)
{
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(mesh.V.rows(), 3);
    for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
        const Eigen::Vector3d v0 = mesh.V.row(mesh.F(f, 0)).head<3>();
        const Eigen::Vector3d v1 = mesh.V.row(mesh.F(f, 1)).head<3>();
        const Eigen::Vector3d v2 = mesh.V.row(mesh.F(f, 2)).head<3>();
        const Eigen::Vector3d face_normal = (v1 - v0).cross(v2 - v0);  // area-weighted
        for (int k = 0; k < 3; ++k) {
            N.row(mesh.F(f, k)) += face_normal.transpose();
        }
    }
    for (Eigen::Index i = 0; i < N.rows(); ++i) {
        const double n = N.row(i).norm();
        if (n > 0.0) N.row(i) /= n;
    }
    return N;
}

/// Build a rendering-only extruded shell from the mid-surface mesh.
///
/// Outer (+h/2 along the vertex normal) and inner (-h/2) surfaces are
/// joined at every boundary edge by a two-triangle quad ribbon. To
/// keep the rim sharp under Polyscope's per-vertex smooth shading,
/// every boundary mid-vertex is **duplicated**: the outer / inner
/// copies used by outer / inner faces stay smooth, and the side-wall
/// ribbon uses a separate set of copies so its perpendicular normals
/// don't average into the outer-surface vertex normals.
///
/// Vertex layout in V_ext:
///   [0, n)            outer copy of every mid-vertex
///   [n, 2*n)          inner copy of every mid-vertex
///   [2*n, ...)        outer + inner duplicates for boundary mid-vertices
///                     (used only by side-wall faces)
///
/// `ext_to_mid[i]` gives the source mid-vertex index for every
/// extruded row, so callers can apply per-mid-vertex displacements /
/// scalars to every extruded copy uniformly.
void build_extruded_mesh(const chladni::mesh::TriMesh& mid,
                         const Eigen::MatrixXd&        normals,
                         double                        thickness,
                         Eigen::MatrixXd&              V_ext,
                         Eigen::MatrixXi&              F_ext,
                         std::vector<Eigen::Index>&    ext_to_mid)
{
    const Eigen::Index n  = mid.V.rows();
    const Eigen::Index nf = mid.F.rows();

    const auto edges = chladni::shell::build_edges(mid.F);
    std::vector<chladni::shell::Edge> boundary;
    std::set<Eigen::Index> bdry_verts;
    for (const auto& e : edges) {
        if (e.is_boundary()) {
            boundary.push_back(e);
            bdry_verts.insert(e.v0);
            bdry_verts.insert(e.v1);
        }
    }

    // Allocate duplicate slots for every boundary mid-vertex.
    // Two duplicates per boundary mid-vertex (one outer, one inner) so
    // both extrusion sides get a crease at the rim.
    std::map<Eigen::Index, Eigen::Index> dup_outer;  // mid -> ext row
    std::map<Eigen::Index, Eigen::Index> dup_inner;
    Eigen::Index next_dup = 2 * n;
    for (auto vi : bdry_verts) {
        dup_outer[vi] = next_dup++;
        dup_inner[vi] = next_dup++;
    }
    const Eigen::Index n_ext = next_dup;

    const double half_h = 0.5 * thickness;

    V_ext.resize(n_ext, 3);
    ext_to_mid.assign(static_cast<std::size_t>(n_ext), -1);

    for (Eigen::Index i = 0; i < n; ++i) {
        V_ext.row(i)     = mid.V.row(i) + half_h * normals.row(i);  // outer
        V_ext.row(n + i) = mid.V.row(i) - half_h * normals.row(i);  // inner
        ext_to_mid[static_cast<std::size_t>(i)]     = i;
        ext_to_mid[static_cast<std::size_t>(n + i)] = i;
    }
    for (auto vi : bdry_verts) {
        V_ext.row(dup_outer[vi]) = mid.V.row(vi) + half_h * normals.row(vi);
        V_ext.row(dup_inner[vi]) = mid.V.row(vi) - half_h * normals.row(vi);
        ext_to_mid[static_cast<std::size_t>(dup_outer[vi])] = vi;
        ext_to_mid[static_cast<std::size_t>(dup_inner[vi])] = vi;
    }

    const Eigen::Index nf_ext =
        2 * nf + 2 * static_cast<Eigen::Index>(boundary.size());
    F_ext.resize(nf_ext, 3);

    // Outer faces — original mid-surface indices, same winding.
    F_ext.topRows(nf) = mid.F;

    // Inner faces — reversed winding, indices shifted by n.
    for (Eigen::Index i = 0; i < nf; ++i) {
        F_ext(nf + i, 0) = static_cast<int>(mid.F(i, 0) + n);
        F_ext(nf + i, 1) = static_cast<int>(mid.F(i, 2) + n);
        F_ext(nf + i, 2) = static_cast<int>(mid.F(i, 1) + n);
    }

    // Side-wall ribbons — use the duplicate vertex copies so the rim
    // crease isn't averaged out by Polyscope's per-vertex shading.
    //
    // Winding matters: build_edges canonicalises boundary edges to
    // (v0 < v1) regardless of how the unique adjacent face traverses
    // them. For a cylinder rim with CCW outer faces, exactly one edge
    // per rim (the wrap-around edge: vertex N back to vertex 0) has
    // its canonical order REVERSED relative to the face winding. If
    // we always use (v0 → v1) here, that one quad gets flipped — its
    // triangle normals point inward, and at the shared duplicate
    // vertices the smooth-shading normal averaging with neighbours
    // produces a visible kink in the rim band (user-reported, 2026-05-12).
    // Pick (a, b) so that (a → b) matches the face's CCW winding.
    for (std::size_t k = 0; k < boundary.size(); ++k) {
        Eigen::Index a, b;
        if (boundary[k].face_left != -1) {
            // The face contains directed edge v0 → v1.
            a = boundary[k].v0;
            b = boundary[k].v1;
        } else {
            // The face contains directed edge v1 → v0.
            a = boundary[k].v1;
            b = boundary[k].v0;
        }
        const Eigen::Index oa = dup_outer[a];
        const Eigen::Index ob = dup_outer[b];
        const Eigen::Index ia = dup_inner[a];
        const Eigen::Index ib = dup_inner[b];
        const Eigen::Index row = 2 * nf + 2 * static_cast<Eigen::Index>(k);
        F_ext(row,     0) = static_cast<int>(oa);
        F_ext(row,     1) = static_cast<int>(ob);
        F_ext(row,     2) = static_cast<int>(ib);
        F_ext(row + 1, 0) = static_cast<int>(oa);
        F_ext(row + 1, 1) = static_cast<int>(ib);
        F_ext(row + 1, 2) = static_cast<int>(ia);
    }
}

// ---------------------------------------------------------------------------
// Mesh-source helpers (scan, geometry rebuild, Polyscope re-registration).
// ---------------------------------------------------------------------------

/// @brief Populate @ref AppState::mesh_choices with every .obj filename
/// in @ref AppState::mesh_dir.
///
/// The earlier version of this scanner filtered by file size (≤ 300 KB)
/// because the FD-on-energy stiffness assembly was @f$O(n^2)@f$ and
/// auto-loading a 25 k-vertex mesh from the dropdown would freeze the
/// UI for tens of seconds. After the analytic-Hessian replacement
/// (Tamstorf–Grinspun 2013, commit `da3023b`) even the largest
/// bundled mesh (rocker-arm_sub.obj, 31 k vertices) recomputes in
/// ~25 s — slow but not surprising. The filter is therefore removed:
/// every @c .obj in @ref AppState::mesh_dir appears in the dropdown,
/// labelled with its on-disk size so the user can predict load time
/// at a glance. Sorted alphabetically by filename for stable ordering.
// Sentinel filenames for procedural meshes — when the dropdown
// selection matches one of these, load_mesh_for_selection generates
// the geometry directly via the chladni::mesh helpers (no .obj file
// involved). Allows the user to play with Chladni patterns on a
// disk / annulus interactively without us having to commit binary
// mesh artifacts to models/.
constexpr const char* kProcDisk      = "*procedural:disk";
constexpr const char* kProcAnnulus   = "*procedural:annulus";
constexpr const char* kProcCylinder  = "*procedural:cylinder";
constexpr const char* kProcIcosphere = "*procedural:icosphere";

void scan_mesh_directory(AppState& app)
{
    app.mesh_choices.clear();
    app.mesh_choice_cstrs.clear();
    app.mesh_choice_filenames.clear();

    // Procedural entries up top.
    app.mesh_choices.emplace_back("* circular disk (procedural)");
    app.mesh_choice_filenames.emplace_back(kProcDisk);
    app.mesh_choices.emplace_back("* annulus (procedural)");
    app.mesh_choice_filenames.emplace_back(kProcAnnulus);
    app.mesh_choices.emplace_back("* cylinder (procedural)");
    app.mesh_choice_filenames.emplace_back(kProcCylinder);
    app.mesh_choices.emplace_back("* icosphere (procedural)");
    app.mesh_choice_filenames.emplace_back(kProcIcosphere);

    if (fs::is_directory(app.mesh_dir)) {
        struct Entry { std::string name; std::uintmax_t size_bytes; };
        std::vector<Entry> entries;
        for (const auto& dir_entry : fs::directory_iterator(app.mesh_dir)) {
            if (!dir_entry.is_regular_file()) continue;
            if (dir_entry.path().extension() != ".obj") continue;
            entries.push_back({
                dir_entry.path().filename().string(),
                dir_entry.file_size()});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.name < b.name; });
        for (const auto& e : entries) {
            const std::uintmax_t kb = (e.size_bytes + 1023) / 1024;
            app.mesh_choices.push_back(
                e.name + " (" + std::to_string(kb) + " KB)");
            app.mesh_choice_filenames.push_back(e.name);
        }
    }

    app.mesh_choice_cstrs.reserve(app.mesh_choices.size());
    for (const auto& s : app.mesh_choices) {
        app.mesh_choice_cstrs.push_back(s.c_str());
    }
}

/// @brief Rebuild every geometric quantity that depends on @c app.mesh.
///
/// Runs the auto-rescale, recomputes per-vertex normals, snapshots the
/// rest configuration, and rebuilds the extruded rendering mesh. Call
/// after replacing @c app.mesh wholesale (e.g. on initial load or when
/// the user picks a different mesh from the dropdown).
void rebuild_geometry_from_mesh(AppState& app)
{
    rescale_to_target_size(app.mesh, 0.20);
    app.vertex_normals = compute_vertex_normals(app.mesh);
    app.V_rest         = app.mesh.V;
    app.n_mid_vertices = app.mesh.V.rows();

    // Build the viz mesh by Loop-subdividing the coarse mesh. With
    // viz_subdivision_passes == 0 we just alias mesh_viz to app.mesh
    // and leave S_viz empty (update_viz_mesh treats this as identity).
    // The subdivision matrix S returned by loop_subdivide_n_times is
    // already shaped for stacked 3D DOFs (3 n_viz x 3 n_coarse) so it
    // applies directly to the modal displacement vector.
    if (app.viz_subdivision_passes > 0) {
        const auto sub = chladni::shell::loop::loop_subdivide_n_times(
            app.mesh.V, app.mesh.F, app.viz_subdivision_passes);
        app.mesh_viz.V = sub.V_sub;
        app.mesh_viz.F = sub.F_sub;
        app.S_viz      = sub.S;
    } else {
        app.mesh_viz   = app.mesh;
        app.S_viz.resize(0, 0);
    }
    app.n_viz_vertices     = app.mesh_viz.V.rows();
    app.vertex_normals_viz = compute_vertex_normals(app.mesh_viz);

    build_extruded_mesh(app.mesh_viz, app.vertex_normals_viz,
                        static_cast<double>(app.h_mm) * 1.0e-3
                          * static_cast<double>(app.viz_thickness_scale),
                        app.V_rest_ext, app.F_ext, app.ext_to_mid);
}

/// @brief (Re)register the extruded shell with Polyscope under the name
/// "shell" and attach the Chladni amplitude scalar quantity.
///
/// Removes any prior "shell" structure first, so the function is idempotent
/// — call it on initial Polyscope setup AND on every subsequent mesh
/// reload. Reads the user's current overlay preferences
/// (@c viz_chladni_enabled, isoline thickness/darkness) so a swap doesn't
/// silently flip those settings back to defaults.
void setup_polyscope_mesh(AppState& app)
{
    polyscope::removeSurfaceMesh("shell", false);

    app.ps_mesh = polyscope::registerSurfaceMesh(
        "shell", app.V_rest_ext, app.F_ext);
    app.ps_mesh->setEdgeWidth(0.0);
    app.ps_mesh->setSmoothShade(true);
    app.ps_mesh->setMaterial("wax");
    app.ps_mesh->setSurfaceColor({0.72F, 0.74F, 0.78F});

    const Eigen::VectorXd zeros =
        Eigen::VectorXd::Zero(app.V_rest_ext.rows());
    app.ps_chladni_q = app.ps_mesh->addVertexScalarQuantity(
        "amplitude (Chladni)", zeros);
    app.ps_chladni_q->setColorMap("coolwarm");
    app.ps_chladni_q->setIsolinesEnabled(true);
    app.ps_chladni_q->setIsolineStyle(polyscope::IsolineStyle::Contour);
    app.ps_chladni_q->setIsolineContourThickness(
        static_cast<double>(app.viz_isoline_thickness));
    app.ps_chladni_q->setIsolineDarkness(
        static_cast<double>(app.viz_isoline_darkness));
    app.ps_chladni_q->setEnabled(app.viz_chladni_enabled);
}

// ---------------------------------------------------------------------------
// Solver invocation (async).
//
// The eigensolve on dense closed meshes takes seconds to minutes —
// long enough to freeze the UI thread (window close ignored, no
// progress feedback). We dispatch onto a detached worker thread so:
//   - The window's close button stays live.
//   - The status line shows "computing..." while in flight.
//   - The next compute starts a new worker even if the previous one
//     is still running (we just drop the in-flight future's result).
//
// poll_recompute() runs every UI frame and harvests the worker's
// result when std::future::wait_for(0) reports ready. On window
// close the AppState destructor releases the future without joining
// the worker (the thread was detached); the OS reaps the worker on
// process exit.
// ---------------------------------------------------------------------------

bool modes_contain_non_finite(const chladni::shell::ShellModes& modes)
{
    if (!modes.omegas.allFinite()) return true;
    if (modes.shapes.size() > 0 && !modes.shapes.allFinite()) return true;
    return false;
}

void apply_modes_result(AppState& app,
                        chladni::shell::ShellModes modes,
                        const ComputeParams& params)
{
    // Defensive: a stale worker that finished after a mesh swap would
    // deliver shapes sized for the OLD mesh. The invalidation path in
    // load_mesh_from_path catches this for direct dropdown swaps, but
    // belt-and-braces it here too — if the result doesn't match the
    // current mesh's DOF count, treat as stale and discard silently.
    if (modes.shapes.size() > 0
        && modes.shapes.rows() != params.expected_dofs)
    {
        std::cerr << "apply_modes_result: discarding stale result (shapes "
                  << modes.shapes.rows() << " rows vs expected "
                  << params.expected_dofs << ")\n";
        return;
    }

    // Defensive: NaN/Inf in the harvested modes (most commonly a
    // numerical breakdown in the Stam path on tricky meshes — see the
    // S.8 caveat in the memory) would propagate through update_viz_mesh
    // to vertex positions and the mesh would render invisible
    // ("mesh disappears"). Surface the error instead.
    if (modes_contain_non_finite(modes)) {
        app.modes_valid = false;
        app.modes       = chladni::shell::ShellModes{};
        app.viz_z.clear();
        app.viz_step.clear();
        if (app.audio != nullptr) {
            std::lock_guard<std::mutex> lk(app.audio->mutex);
            app.audio->modes.clear();
        }
        app.last_status =
            std::string{"error: compute returned NaN/Inf ["}
          + params.assembly_label + "] — try toggling Stam or n_passes";
        std::cerr << "apply_modes_result: NaN/Inf in result ["
                  << params.assembly_label << "] — rejected\n";
        return;
    }

    const std::size_t n_modes = static_cast<std::size_t>(modes.omegas.size());
    app.modes = std::move(modes);
    app.modes_valid = true;
    app.last_status = std::string{"computed "} + std::to_string(n_modes)
                    + " modes [" + params.assembly_label + "]";

    // Adapt the audio band-pass slider range and defaults to the
    // freshly computed spectrum.
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double f_min = app.modes.omegas(0) / two_pi;
    const double f_max =
        app.modes.omegas(app.modes.omegas.size() - 1) / two_pi;
    // Floor the spectrum minimum at the source: spectrum_min_hz is the
    // v_min of the two logarithmic "Low/High cut (Hz)" sliders below, and a
    // rigid-body mode slipping past the spectrum filter (acknowledged
    // possible elsewhere) makes omegas(0)≈0 → f_min≈0 → log-slider v_min=0,
    // the same pitfall reviews F3/G2 fixed for the hardcoded sliders.
    app.spectrum_min_hz   = std::max(1.0e-3F, static_cast<float>(f_min));
    // spectrum_max_hz is the v_max of those same logarithmic sliders; if the
    // whole spectrum collapses toward 0 (f_max < the v_min floor) an ImGui
    // log slider with v_min > v_max misbehaves, so keep v_max >= v_min
    // (review7 I7 — the unswept upper half of the F3/G2/H2 slider fix).
    app.spectrum_max_hz   = std::max(app.spectrum_min_hz,
                                     static_cast<float>(f_max));
    app.audio_low_cut_hz  = app.spectrum_min_hz;
    const double nyquist_hz =
        (app.audio != nullptr) ? 0.5 * app.audio->sample_rate : 22050.0;
    app.audio_high_cut_hz = std::min(
        app.spectrum_max_hz, static_cast<float>(nyquist_hz));

    // Spectrum diagnostic — uses the captured ComputeParams, NOT
    // app.E_GPa / app.h_mm / etc. The user may have dragged sliders
    // between kick_off and harvest; logging the live values made the
    // output deeply misleading (an "h=0.28 mm" line could appear next
    // to the h=1 mm spectrum the worker actually computed).
    std::cout << "recompute_modes("
              << "E="        << params.E_GPa << " GPa"
              << ", nu="     << params.nu
              << ", rho="    << params.rho
              << ", h="      << params.h_mm << " mm"
              << ", n_modes=" << params.n_modes_request
              << ", n_passes=" << params.loop.n_passes
              << ", assembly=" << params.assembly_label
              << ")\n  spectrum (Hz):";
    const Eigen::Index n_show =
        std::min<Eigen::Index>(12, app.modes.omegas.size());
    for (Eigen::Index k = 0; k < n_show; ++k) {
        std::cout << " " << (app.modes.omegas(k) / two_pi);
    }
    std::cout << " ... max=" << f_max << "\n";

    // Reset the viz state to match the new spectrum (size + steps).
    const std::size_t M = static_cast<std::size_t>(app.modes.omegas.size());
    app.viz_z.assign(M, std::complex<double>{0.0, 0.0});
    app.viz_step.assign(M, std::complex<double>{1.0, 0.0});

    if (app.audio == nullptr) return;

    // Push the new bank (with Rayleigh damping baked in) to the audio thread.
    const double slowdown = std::max(1.0e-3, static_cast<double>(app.viz_slowdown));
    const double dt    = (1.0 / app.audio->sample_rate) / slowdown;
    const double alpha = static_cast<double>(app.alpha);
    const double beta  = static_cast<double>(app.beta);

    std::lock_guard<std::mutex> lk(app.audio->mutex);
    app.audio->modes.assign(static_cast<std::size_t>(app.modes.omegas.size()),
                            AudioMode{});
    constexpr double pi = std::numbers::pi_v<double>;
    for (Eigen::Index k = 0; k < app.modes.omegas.size(); ++k) {
        const double w = app.modes.omegas(k);
        const double d = 0.5 * (alpha + beta * w * w);
        const double decay = std::exp(-d * dt);
        const std::complex<double> rot{std::cos(w * dt), std::sin(w * dt)};

        auto& am = app.audio->modes[static_cast<std::size_t>(k)];
        am.angular_frequency = w;
        am.damping_rate      = d;
        am.step              = decay * rot;
        am.z                 = {0.0, 0.0};  // silent until struck
        am.audible           = (w * dt < pi);  // below audio Nyquist
    }
}

/// Dispatch a modal-eigensolve callable onto a detached worker thread and
/// wire its future into @c app so @ref poll_recompute collects the result on
/// a later frame. Shared by the primary recompute and the legacy fallback so
/// neither ever runs the (potentially slow) eigensolve on the UI thread.
/// packaged_task + detached thread (not std::async) keeps the future
/// destructor non-blocking on the window-close path.
template <typename Fn>
void dispatch_modal_task(AppState& app, Fn&& fn)
{
    auto task = std::make_shared<
        std::packaged_task<chladni::shell::ShellModes()>>(std::forward<Fn>(fn));
    app.compute_future    = task->get_future();
    app.compute_in_flight = true;
    app.last_status = std::string{"computing... ["}
                    + app.compute_params.assembly_label + "]";
    std::thread([task]() { (*task)(); }).detach();
}

void kick_off_recompute(AppState& app)
{
    // Snapshot every input the worker will use. All subsequent slider
    // edits (h_mm, E_GPa, n_passes, use_stam, ...) won't leak into
    // the running compute and won't pollute the eventual diagnostic
    // log: apply_modes_result reads from app.compute_params, not the
    // live AppState fields.
    ComputeParams p;
    p.E_GPa           = app.E_GPa;
    p.nu              = app.nu;
    p.rho             = app.rho;
    p.h_mm            = app.h_mm;
    p.n_modes_request = std::max(1, app.n_modes_request);
    p.assembler       = app.assembler;
    p.loop            = app.loop_params;
    p.lme             = app.lme_params;
    switch (p.assembler) {
    case AssemblerKind::Loop:
        p.assembly_label = p.loop.use_stam ? "Loop+Stam" : "Loop";
        break;
    case AssemblerKind::LME:
        p.assembly_label = p.lme.use_second_order_sme
                               ? "SME (Rosolen-Millan 2013)"
                               : "LME (Arroyo-Ortiz)";
        break;
    case AssemblerKind::Legacy:
        p.assembly_label = "Legacy (CST+IBM)";
        break;
    }
    p.expected_dofs   = 3 * app.mesh.V.rows();
    app.compute_params = p;

    const chladni::IsotropicMaterial mat{
        .youngs_modulus = static_cast<double>(p.E_GPa) * 1.0e9,
        .poisson_ratio  = static_cast<double>(p.nu),
        .density        = static_cast<double>(p.rho),
    };
    const double thickness = static_cast<double>(p.h_mm) * 1.0e-3;
    const std::size_t n_modes = static_cast<std::size_t>(p.n_modes_request);
    const Eigen::MatrixXd V = app.mesh.V;
    const Eigen::MatrixXi F = app.mesh.F;
    const auto assembler_kind = p.assembler;
    const auto loop_params = p.loop;
    const auto lme_params = p.lme;

    dispatch_modal_task(
        app,
        [V, F, mat, thickness, n_modes, assembler_kind, loop_params, lme_params]() {
            const auto sm = chladni::shell::shell_material_from_isotropic(
                mat, thickness);
            switch (assembler_kind) {
            case AssemblerKind::Loop: {
                chladni::shell::LoopAssembler assembler{loop_params};
                return chladni::shell::compute_shell_modes(
                    V, F, mat, sm, thickness, n_modes, assembler);
            }
            case AssemblerKind::LME: {
                chladni::shell::LMEAssembler assembler{lme_params};
                return chladni::shell::compute_shell_modes(
                    V, F, mat, sm, thickness, n_modes, assembler);
            }
            case AssemblerKind::Legacy:
            default:
                return chladni::shell::compute_shell_modes(
                    V, F, mat, thickness, n_modes);
            }
        });
}

void poll_recompute(AppState& app)
{
    if (!app.compute_in_flight) return;
    if (!app.compute_future.valid())  {
        app.compute_in_flight = false;
        return;
    }
    if (app.compute_future.wait_for(std::chrono::seconds(0))
        != std::future_status::ready)
    {
        return;
    }

    try {
        auto modes = app.compute_future.get();
        apply_modes_result(app, std::move(modes), app.compute_params);
    } catch (const std::exception& e) {
        // Modern-assembler path failed — try the legacy fallback
        // synchronously on the UI thread (legacy is fast enough that this
        // doesn't re-freeze the window in practice). Reuse the captured
        // ComputeParams so the fallback's eventual log/audio match what
        // the user originally asked for. After 9.8 the LME path accepts
        // both flat and curved meshes, so this fallback now mostly
        // catches genuine assembler errors (e.g. valence-2 strip
        // corners on the Loop path) rather than mesh-shape mismatches.
        if (app.compute_params.assembler != AssemblerKind::Legacy) {
            std::cerr << "recompute_modes: "
                      << app.compute_params.assembly_label
                      << " path failed (" << e.what()
                      << ") — falling back to legacy CST+IBM\n";
            // Re-dispatch the legacy solve on a WORKER thread. It used to run
            // synchronously here on the UI thread, freezing the window the
            // async design exists to keep responsive. Mark compute_params as
            // Legacy so that if the fallback ITSELF throws, the next poll
            // reports the error (the else branch) instead of looping back
            // into another fallback.
            const chladni::IsotropicMaterial mat{
                .youngs_modulus =
                    static_cast<double>(app.compute_params.E_GPa) * 1.0e9,
                .poisson_ratio  =
                    static_cast<double>(app.compute_params.nu),
                .density        =
                    static_cast<double>(app.compute_params.rho),
            };
            const double thickness =
                static_cast<double>(app.compute_params.h_mm) * 1.0e-3;
            const std::size_t n_modes = static_cast<std::size_t>(
                app.compute_params.n_modes_request);
            const Eigen::MatrixXd V = app.mesh.V;
            const Eigen::MatrixXi F = app.mesh.F;
            app.compute_params.assembler     = AssemblerKind::Legacy;
            app.compute_params.assembly_label =
                "Legacy fallback after assembler error";
            dispatch_modal_task(app, [V, F, mat, thickness, n_modes]() {
                return chladni::shell::compute_shell_modes(
                    V, F, mat, thickness, n_modes);
            });
            return;  // keep compute_in_flight = true; next poll collects it
        }
        app.modes_valid = false;
        app.last_status = std::string{"error: "} + e.what();
        std::cerr << "recompute_modes: " << e.what() << "\n";
    }
    app.compute_in_flight = false;
}

/// @brief Load an OBJ from disk and reinstall it as the active mesh.
///
/// On success: replaces @c app.mesh, runs @ref rebuild_geometry_from_mesh,
/// re-registers the Polyscope structure, recentres the camera, and
/// recomputes the modal eigensolve. On failure (file missing, parse
/// error, etc.) the previous mesh is left in place and a one-line
/// message goes into @c app.last_status. Safe to call repeatedly from
/// the UI thread.
///
/// @param app   Application state.
/// @param path  Filesystem path to a .obj file.
/// @return true on success, false otherwise.
// Common install-a-new-mesh pipeline, shared between OBJ loading and
// procedural generation. Replaces app.mesh, invalidates the previous
// mesh's modes / viz state / audio bank (so a stale shapes matrix can't
// be indexed with the new mesh's vertex IDs), re-registers polyscope,
// and kicks off the async recompute.
void install_new_mesh(AppState& app,
                      chladni::mesh::TriMesh new_mesh,
                      std::string display_name)
{
    app.mesh = std::move(new_mesh);
    app.current_mesh_name = std::move(display_name);
    rebuild_geometry_from_mesh(app);

    app.modes_valid = false;
    app.modes       = chladni::shell::ShellModes{};
    app.viz_z.clear();
    app.viz_step.clear();
    if (app.audio != nullptr) {
        std::lock_guard<std::mutex> lk(app.audio->mutex);
        app.audio->modes.clear();
    }

    std::cout << "strike_gui: loaded '" << app.current_mesh_name
              << "' (" << app.mesh.V.rows() << "v / "
              << app.mesh.F.rows() << "f); extruded "
              << app.V_rest_ext.rows() << "v / "
              << app.F_ext.rows() << "f\n";

    setup_polyscope_mesh(app);
    polyscope::view::resetCameraToHomeView();
    kick_off_recompute(app);
}

bool load_mesh_from_path(AppState& app, const fs::path& path)
{
    chladni::mesh::TriMesh new_mesh;
    try {
        new_mesh = chladni::mesh::load_obj(path);
    } catch (const std::exception& e) {
        app.last_status = std::string{"load failed: "} + e.what();
        std::cerr << "load_mesh_from_path: " << e.what() << "\n";
        return false;
    }
    install_new_mesh(app, std::move(new_mesh), path.filename().string());
    return true;
}

// Dispatch a procedural-mesh sentinel filename to the appropriate
// generator. Defaults chosen for moderate-resolution Chladni-pattern
// visualisation; the GUI's 20cm auto-rescale normalises the size.
bool load_procedural_mesh(AppState& app, const std::string& kind)
{
    chladni::mesh::TriMesh new_mesh;
    std::string display_name;
    // Checkerboard triangulation needs an even azimuthal count (seam
    // closure); round the slider value down to even rather than
    // erroring. Consistent and UnionJack take any count.
    const auto split = app.proc_quad_split;
    auto even_if_alt = [split](int n) {
        return split == chladni::mesh::QuadSplit::Checkerboard
                   ? std::max(4, n & ~1)
                   : n;
    };
    const char* alt_tag =
        split == chladni::mesh::QuadSplit::UnionJack      ? " (uj)"
        : split == chladni::mesh::QuadSplit::Checkerboard ? " (cb)"
                                                          : "";
    try {
        if (kind == kProcDisk) {
            // Polar disk: highest discrete rotational symmetry of the
            // candidate topologies we surveyed (D_n at n = n_az). The
            // hex disk gives a cleaner central area but its D_6
            // symmetry aliases high-m doublets like (7,0) into m'=1
            // admixture (curved nodal lines). The iso disk
            // (cumulative-phase concentric rings) has no useful
            // symmetry and corrupts modes globally. See
            // project_chladni_disk_mesh_dependence for the full
            // analysis. Resolution comes from AppState sliders so the
            // user can adjust without recompiling.
            const int n_az  = even_if_alt(std::max(3, app.proc_disk_n_az));
            const int n_rad = std::max(1, app.proc_disk_n_rad);
            new_mesh = chladni::mesh::generate_circular_disk(
                /*R=*/0.10, n_az, n_rad, split);
            display_name = "polar disk " + std::to_string(n_az)
                         + "x" + std::to_string(n_rad) + alt_tag;
        } else if (kind == kProcAnnulus) {
            // Resolution and b/a ratio come from AppState sliders.
            // b/a = 0.5 matches Leissa Table 2.18 reference column.
            const int n_az  = even_if_alt(std::max(3, app.proc_annulus_n_az));
            const int n_rad = std::max(2, app.proc_annulus_n_rad);
            const double b_over_a = std::clamp(
                static_cast<double>(app.proc_annulus_b_over_a), 0.05, 0.95);
            const double R_out = 0.10;
            const double R_in  = R_out * b_over_a;
            new_mesh = chladni::mesh::generate_annulus(
                R_out, R_in, n_az, n_rad, split);
            char ratio_buf[16];
            std::snprintf(ratio_buf, sizeof(ratio_buf), "%.2f", b_over_a);
            display_name = "annulus b/a=" + std::string(ratio_buf)
                         + "  " + std::to_string(n_az)
                         + "x" + std::to_string(n_rad) + alt_tag;
        } else if (kind == kProcCylinder) {
            // Resolution comes from AppState. The 2026-05-14 cylinder
            // convergence audit found doublet-split chirality bias
            // drops below 0.1% only at ~96 around x 32 along (3168 V);
            // coarser meshes show visibly skewed Chladni patterns
            // that are mesh-discretization artifacts, not FEM bugs.
            const int n_around =
                even_if_alt(std::max(3, app.proc_cylinder_n_around));
            const int n_along  = std::max(1, app.proc_cylinder_n_along);
            new_mesh = chladni::mesh::generate_cylinder(
                /*R=*/0.10, /*L=*/0.20, n_around, n_along, split);
            display_name = "cylinder " + std::to_string(n_around)
                         + "x" + std::to_string(n_along) + alt_tag;
        } else if (kind == kProcIcosphere) {
            // Closed sphere via icosahedron + edge-midpoint subdivision
            // (see generate_icosphere docstring). k=2 → 162V / 320F,
            // k=3 → 642V / 1280F. Primary use is visual verification
            // of the curved-shell LME path (§10 step 9.x) on a closed,
            // boundary-free shell.
            const int k = std::max(0, app.proc_icosphere_subdivisions);
            new_mesh = chladni::mesh::generate_icosphere(
                /*radius=*/0.10, k);
            display_name = "icosphere k=" + std::to_string(k);
        } else {
            throw std::invalid_argument(
                "unknown procedural mesh kind: " + kind);
        }
    } catch (const std::exception& e) {
        app.last_status = std::string{"procedural mesh failed: "}
                        + e.what();
        std::cerr << "load_procedural_mesh: " << e.what() << "\n";
        return false;
    }
    install_new_mesh(app, std::move(new_mesh), std::move(display_name));
    return true;
}

// Single entry point used by the dropdown / drop-handler. Dispatches
// procedural sentinels to load_procedural_mesh, everything else to
// load_mesh_from_path.
bool load_mesh_for_selection(AppState& app, const std::string& filename)
{
    if (!filename.empty() && filename.front() == '*') {
        return load_procedural_mesh(app, filename);
    }
    return load_mesh_from_path(app, app.mesh_dir / filename);
}

/// Re-derive every audio mode's per-sample step from its stored
/// (omega, damping_rate) using the current slowdown. Call after the
/// slowdown slider changes so audio pitch tracks viz speed.
void refresh_audio_slowdown(AppState& app)
{
    if (app.audio == nullptr) return;
    const double slowdown = std::max(1.0e-3, static_cast<double>(app.viz_slowdown));
    const double dt       = (1.0 / app.audio->sample_rate) / slowdown;
    std::lock_guard<std::mutex> lk(app.audio->mutex);
    for (auto& m : app.audio->modes) {
        const double decay = std::exp(-m.damping_rate * dt);
        const std::complex<double> rot{std::cos(m.angular_frequency * dt),
                                       std::sin(m.angular_frequency * dt)};
        m.step    = decay * rot;
        m.audible = (m.angular_frequency * dt < std::numbers::pi_v<double>);
    }
}

/// @brief Re-derive every audio mode's @c damping_rate and per-sample
/// @c step from the current @c app.alpha / @c app.beta sliders.
///
/// The Rayleigh damping per-mode rate is
///   d_i = (alpha + beta * omega_i^2) / 2
/// and the per-sample @c step is @c exp(-d * dt) * cis(omega * dt) at
/// the current slowdown-dilated dt. Without this refresh, alpha and
/// beta only take effect on the next Recompute — touching the sliders
/// and reverting them would leave the audio bank with whatever damping
/// happened to be in place when Recompute was last pressed, which
/// silently kills the sound on stiffer materials and feels like a bug.
void refresh_audio_damping(AppState& app)
{
    if (app.audio == nullptr) return;
    const double alpha    = static_cast<double>(app.alpha);
    const double beta     = static_cast<double>(app.beta);
    const double slowdown =
        std::max(1.0e-3, static_cast<double>(app.viz_slowdown));
    const double dt = (1.0 / app.audio->sample_rate) / slowdown;
    std::lock_guard<std::mutex> lk(app.audio->mutex);
    for (auto& m : app.audio->modes) {
        const double w     = m.angular_frequency;
        const double d     = 0.5 * (alpha + beta * w * w);
        const double decay = std::exp(-d * dt);
        const std::complex<double> rot{std::cos(w * dt), std::sin(w * dt)};
        m.damping_rate = d;
        m.step         = decay * rot;
        m.audible      = (w * dt < std::numbers::pi_v<double>);
    }
}

void apply_strike(AppState& app, Eigen::Index vertex_idx)
{
    if (!app.modes_valid || app.audio == nullptr) return;
    if (vertex_idx < 0 || vertex_idx >= app.mesh.V.rows()) return;
    if (app.modes.omegas.size() == 0) return;

    // Inward normal at the clicked vertex (force points into the shell).
    const double scale = static_cast<double>(app.hammer_force)
                       * static_cast<double>(app.strike_intensity);
    const Eigen::Vector3d f =
        -app.vertex_normals.row(vertex_idx).transpose() * scale;

    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    const double omega_hammer = tau * static_cast<double>(app.hammer_cutoff_hz);
    const double omega_low    = tau * static_cast<double>(app.audio_low_cut_hz);
    const double omega_high   = tau * static_cast<double>(app.audio_high_cut_hz);

    double max_amp = 0.0;
    double max_dot = 0.0;
    double min_H_band = 1.0;
    double min_H_hammer = 1.0;
    Eigen::Index argmax_amp = -1;
    std::vector<double> initial_amps(static_cast<std::size_t>(app.modes.omegas.size()));
    for (Eigen::Index k = 0; k < app.modes.omegas.size(); ++k) {
        const double dot = app.modes.shapes(3 * vertex_idx + 0, k) * f.x()
                         + app.modes.shapes(3 * vertex_idx + 1, k) * f.y()
                         + app.modes.shapes(3 * vertex_idx + 2, k) * f.z();
        max_dot = std::max(max_dot, std::abs(dot));
        const double w = app.modes.omegas(k);

        // Hammer roll-off (low-pass): finite contact spectrum.
        const double H_hammer = 1.0 / std::sqrt(1.0 + (w * w) / (omega_hammer * omega_hammer));
        min_H_hammer = std::min(min_H_hammer, H_hammer);

        // Audio band-pass at the mode's frequency. First-order high-
        // and low-pass envelopes; product is the band-pass shape.
        const double H_high_pass = (omega_low > 0.0)
            ? 1.0 / std::sqrt(1.0 + (omega_low * omega_low) / (w * w))
            : 1.0;
        const double H_low_pass  = 1.0 / std::sqrt(1.0 + (w * w) / (omega_high * omega_high));

        const double H_band = H_high_pass * H_low_pass;
        min_H_band = std::min(min_H_band, H_band);

        // Initial mode coordinate from impulsive force: q'_0 = phi^T f / 1
        // (mass-orthonormal phi). Sustained sinusoid amplitude = q'_0 / omega.
        // Guard ω≈0: a rigid-body mode slipping past the spectrum filter would
        // divide the impulse by zero and blow the amplitude up.
        const double a = (w > 1.0e-6) ? (dot / w) * H_hammer * H_band : 0.0;
        initial_amps[static_cast<std::size_t>(k)] = a;
        if (std::abs(a) > std::abs(max_amp)) {
            argmax_amp = k;
        }
        max_amp = std::max(max_amp, std::abs(a));
    }

    // Verbose strike breakdown so the user can see where attenuation is
    // happening when audio mysteriously goes silent. min H_hammer / min
    // H_band collapsing toward 0 means the slider settings are nuking
    // the whole bank; max_dot near 0 means the strike point is on a
    // node of every audible mode.
    const double f_argmax = (argmax_amp >= 0)
        ? app.modes.omegas(argmax_amp) / tau : 0.0;
    std::cout << "strike v=" << vertex_idx
              << " force=" << app.hammer_force
              << " cutoff=" << app.hammer_cutoff_hz << " Hz"
              << " intensity=" << app.strike_intensity
              << " low_cut=" << app.audio_low_cut_hz
              << " high_cut=" << app.audio_high_cut_hz
              << " max_dot=" << max_dot
              << " min_H_hammer=" << min_H_hammer
              << " min_H_band=" << min_H_band
              << " max_modal_amp=" << max_amp
              << " (peak mode " << argmax_amp << " @ " << f_argmax << " Hz)\n";

    {
        std::lock_guard<std::mutex> lk(app.audio->mutex);
        for (std::size_t k = 0; k < initial_amps.size(); ++k) {
            // Don't pour strike energy into a supra-Nyquist mode — the audio
            // callback skips it, so its state would just sit there stale.
            if (!app.audio->modes[k].audible) continue;
            // Setting z = (A, 0) gives Im(z(0)) = 0 and the imaginary part
            // evolves as A * exp(-d t) * sin(omega t), the desired response.
            app.audio->modes[k].z = std::complex<double>(initial_amps[k], 0.0);
        }
    }

    // Mirror to viz state (no mutex — viz is main-thread only).
    if (app.viz_z.size() == initial_amps.size()) {
        for (std::size_t k = 0; k < initial_amps.size(); ++k) {
            app.viz_z[k] = std::complex<double>(initial_amps[k], 0.0);
        }
    }

    ++app.strike_count;
}

/// @brief Drive a single normal mode and silence every other.
///
/// Sets the modal coordinate of mode @p k_one_based to (@p amp, 0) and
/// zeroes all others, both in the audio mode bank and in the viz state.
/// The audio callback emits @c Im(z), which evolves as
/// @c amp*exp(-d*t)*sin(omega*t) — a pure sinusoid at the modal
/// frequency. With damping near zero (alpha and beta both pushed down)
/// the rendered deformation freezes into the textbook STATIONARY
/// Chladni pattern: nodal lines stand still while antinodes pulse.
///
/// @param app          Application state.
/// @param k_one_based  Mode index, 1-based; clamped into [1, n_modes].
/// @param amp          Modal coordinate magnitude. Roughly comparable
///                     to the per-mode amplitude after a strike (which
///                     is typically O(1e-2) — see strike's
///                     max_modal_amp print).
void ring_single_mode(AppState& app, int k_one_based, double amp)
{
    if (!app.modes_valid) return;
    if (app.modes.omegas.size() == 0) return;
    const int n = static_cast<int>(app.modes.omegas.size());
    const int k = std::clamp(k_one_based, 1, n) - 1;

    constexpr double tau = 2.0 * std::numbers::pi_v<double>;
    const double f_k = app.modes.omegas(k) / tau;
    std::cout << "ring mode " << (k + 1) << " @ " << f_k
              << " Hz, amp=" << amp
              << (app.single_mode_sound ? " (audio on)" : " (audio off)")
              << "\n";

    // Audio bank: when single_mode_sound is off, keep every audio
    // resonator silent — the user wanted a *visual* Chladni demo only.
    // When on, write amp to mode k and zero the rest.
    if (app.audio != nullptr) {
        std::lock_guard<std::mutex> lk(app.audio->mutex);
        for (std::size_t i = 0; i < app.audio->modes.size(); ++i) {
            if (app.single_mode_sound) {
                app.audio->modes[i].z = (static_cast<int>(i) == k)
                    ? std::complex<double>(amp, 0.0)
                    : std::complex<double>(0.0, 0.0);
            } else {
                app.audio->modes[i].z = std::complex<double>(0.0, 0.0);
            }
        }
    }
    // Visualization always reflects single-mode state — that's the
    // whole point of the panel, independent of the sound toggle.
    if (app.viz_z.size() == static_cast<std::size_t>(n)) {
        for (int i = 0; i < n; ++i) {
            app.viz_z[static_cast<std::size_t>(i)] = (i == k)
                ? std::complex<double>(amp, 0.0)
                : std::complex<double>(0.0, 0.0);
        }
    }
    ++app.strike_count;
}

// ---------------------------------------------------------------------------
// Slow-motion deformation viz.
// ---------------------------------------------------------------------------

/// Rebuild the rendered extruded shell from the mid-surface mesh and
/// the current physical thickness h * viz_thickness_scale, then push
/// the new vertex positions to Polyscope. Cheap (a few hundred
/// vertices); fine to call on every slider tick.
void rebuild_render_extrusion(AppState& app)
{
    // Extrude from the *viz* mesh (Loop-subdivided copy of the coarse
    // FEM mesh) so the rendered surface always carries the same
    // refinement level the chladni overlay assumes.
    build_extruded_mesh(app.mesh_viz, app.vertex_normals_viz,
                        static_cast<double>(app.h_mm) * 1.0e-3
                          * static_cast<double>(app.viz_thickness_scale),
                        app.V_rest_ext, app.F_ext, app.ext_to_mid);
    if (app.ps_mesh != nullptr) {
        app.ps_mesh->updateVertexPositions(app.V_rest_ext);
    }
}

/// Apply current viz state to ps_mesh. The Polyscope mesh is the
/// extruded shell (outer + inner + side walls); modal displacement
/// from the mid-surface is applied to BOTH the outer and inner copy
/// of every mid-surface vertex, preserving the wall thickness under
/// small-strain deformation.
void update_viz_mesh(AppState& app, double frame_dt)
{
    // Defensive: in addition to the modes_valid gate, also check that
    // the modes' shape matrix actually matches the current mesh's DOF
    // count. apply_modes_result rejects mismatched / NaN results
    // already, but if anything sneaks through, fall back to showing
    // the rest pose rather than rendering a mesh full of NaN
    // positions (which polyscope renders as invisible — the user
    // reported "mesh disappears").
    const Eigen::Index expected_rows = 3 * app.n_mid_vertices;
    const bool shapes_ok =
           app.modes.shapes.size() > 0
        && app.modes.shapes.rows() == expected_rows;
    if (!app.viz_enabled || !app.modes_valid || !shapes_ok) {
        if (app.ps_mesh != nullptr) {
            app.ps_mesh->updateVertexPositions(app.V_rest_ext);
        }
        return;
    }

    const double slowdown = std::max(1.0, static_cast<double>(app.viz_slowdown));
    const double viz_dt   = frame_dt / slowdown;

    // Recompute per-mode step at the current viz_dt.
    for (Eigen::Index k = 0; k < app.modes.omegas.size(); ++k) {
        const double w = app.modes.omegas(k);
        const double d = (app.audio != nullptr && static_cast<std::size_t>(k) < app.audio->modes.size())
            ? app.audio->modes[static_cast<std::size_t>(k)].damping_rate
            : 0.5 * (static_cast<double>(app.alpha) + static_cast<double>(app.beta) * w * w);
        const double decay = std::exp(-d * viz_dt);
        const std::complex<double> rot{std::cos(w * viz_dt), std::sin(w * viz_dt)};
        app.viz_step[static_cast<std::size_t>(k)] = decay * rot;
    }

    // Step viz state by one (slowed) sample.
    for (std::size_t k = 0; k < app.viz_z.size(); ++k) {
        app.viz_z[k] *= app.viz_step[k];
    }

    // Compose the coarse mid-surface deformation u_coarse =
    // sum_k Im(viz_z_k) * phi_k. The eigenvectors phi_k live on the
    // coarse FEM mesh (length 3 n_mid_vertices).
    Eigen::VectorXd u_coarse = Eigen::VectorXd::Zero(3 * app.n_mid_vertices);
    for (Eigen::Index k = 0; k < app.modes.omegas.size(); ++k) {
        u_coarse += app.viz_z[static_cast<std::size_t>(k)].imag()
                  * app.modes.shapes.col(k);
    }
    u_coarse *= static_cast<double>(app.viz_amplification);

    // Push to the viz (subdivided) mesh. S_viz is shaped 3 n_viz x
    // 3 n_coarse for stacked DOFs; with viz_subdivision_passes == 0 it's
    // empty and we just alias u to u_coarse.
    Eigen::VectorXd u;
    if (app.viz_subdivision_passes > 0 && app.S_viz.rows() > 0) {
        u.noalias() = app.S_viz * u_coarse;
    } else {
        u = u_coarse;
    }

    // Apply the per-viz-vertex displacement to every extruded copy
    // (outer, inner, AND boundary duplicates) using ext_to_mid. The
    // same loop computes the signed normal displacement scalar for
    // the Chladni overlay; copies of the same viz-vertex carry the
    // same scalar so the field is continuous across the rim crease.
    const double inv_amp = 1.0 / static_cast<double>(app.viz_amplification);

    // Per-viz-vertex signed normal displacement (raw, un-amplified).
    // Computed against the viz mesh's normals — for a Loop-subdivided
    // mesh these are smoother than the coarse mesh's normals, but at
    // passes == 0 they coincide.
    Eigen::VectorXd s_per_mid = Eigen::VectorXd::Zero(app.n_viz_vertices);
    for (Eigen::Index i = 0; i < app.n_viz_vertices; ++i) {
        const Eigen::Vector3d u_i{u(3 * i + 0) * inv_amp,
                                  u(3 * i + 1) * inv_amp,
                                  u(3 * i + 2) * inv_amp};
        const Eigen::Vector3d n_i = app.vertex_normals_viz.row(i).head<3>();
        s_per_mid(i) = u_i.dot(n_i);
    }

    Eigen::MatrixXd V = app.V_rest_ext;
    Eigen::VectorXd chladni_scalar(V.rows());
    for (Eigen::Index ei = 0; ei < V.rows(); ++ei) {
        const Eigen::Index mid =
            app.ext_to_mid[static_cast<std::size_t>(ei)];
        V(ei, 0) += u(3 * mid + 0);
        V(ei, 1) += u(3 * mid + 1);
        V(ei, 2) += u(3 * mid + 2);
        chladni_scalar(ei) = s_per_mid(mid);
    }
    if (app.ps_mesh != nullptr) {
        app.ps_mesh->updateVertexPositions(V);
    }

    if (app.viz_chladni_enabled && app.ps_chladni_q != nullptr) {
        const double max_s = chladni_scalar.cwiseAbs().maxCoeff();
        if (max_s > 1.0e-12) {
            // Polyscope's CONTOUR shader draws lines where
            // fract(|value/period|) = 0.5 — i.e. at +/-period/2,
            // +/-3*period/2, ... NOT at 0. To put a contour AT the zero
            // level (the actual nodal set we care about), shift the
            // values by +max_s so the original 0 lands at max_s, which
            // with period = 2*max_s/n_iso lies exactly where the shader
            // draws a line. The map range shifts in lockstep so the
            // diverging colormap still places white at original 0.
            const double n_iso = std::max(1.0, static_cast<double>(app.viz_isoline_count));
            const double period = 2.0 * max_s / n_iso;
            Eigen::VectorXd shifted = chladni_scalar.array() + max_s;
            // Noise-floor suppression: where local |amplitude| is below
            // viz_chladni_noise_floor * max_s, replace the rendered
            // scalar with a value at integer multiples of the period
            // (fract == 0, far from contour position 0.5). This keeps
            // bulk nodal lines untouched while killing the fractal
            // pseudo-contours that appear near the centre of high-m
            // modes, where the analytic amplitude r^m drops below the
            // FEM / Loop-basis truncation noise.
            const float nf = app.viz_chladni_noise_floor;
            if (nf > 0.0F) {
                const double thr = static_cast<double>(nf) * max_s;
                // Sentinel = 2*max_s lands at fract(2*max_s/period) =
                // fract(n_iso) = 0 for any integer n_iso — guaranteed
                // off-contour. For non-integer n_iso users we land
                // between bands. Colormap value 2*max_s is the "max
                // red" endpoint; the suppressed region therefore reads
                // as a solid colour patch rather than a noisy stipple.
                const double sentinel = 2.0 * max_s;
                for (Eigen::Index i = 0; i < chladni_scalar.size(); ++i) {
                    if (std::abs(chladni_scalar(i)) < thr) {
                        shifted(i) = sentinel;
                    }
                }
            }
            app.ps_chladni_q->updateData(shifted);
            app.ps_chladni_q->setMapRange({0.0, 2.0 * max_s});
            app.ps_chladni_q->setIsolinePeriod(period, /*isRelative=*/false);
            app.ps_chladni_q->setIsolineContourThickness(
                static_cast<double>(app.viz_isoline_thickness));
            app.ps_chladni_q->setIsolineDarkness(
                static_cast<double>(app.viz_isoline_darkness));
        }
    }
}

// ---------------------------------------------------------------------------
// miniaudio playback callback.
// ---------------------------------------------------------------------------

void audio_callback(ma_device* device, void* output, const void* /*input*/,
                    ma_uint32 frame_count)
{
    auto* state = static_cast<AudioState*>(device->pUserData);
    auto* out   = static_cast<float*>(output);
    // The device may not honour the requested mono layout; fill every channel.
    const ma_uint32 channels = device->playback.channels;

    std::unique_lock<std::mutex> lk(state->mutex, std::try_to_lock);
    if (!lk.owns_lock()) {
        std::memset(out, 0,
                    static_cast<std::size_t>(frame_count) * channels * sizeof(float));
        return;
    }

    constexpr float  gain         = 0.3F;
    constexpr double denorm_floor = 1.0e-15;  // flush decayed modes to silence
    for (ma_uint32 i = 0; i < frame_count; ++i) {
        double sample = 0.0;
        for (auto& m : state->modes) {
            // Skip supra-Nyquist modes: stepping them would alias a spurious
            // low pitch into the output. They are kept in the bank (and the
            // visualiser) but contribute nothing to the audio sum.
            if (!m.audible) continue;
            sample += m.z.imag();
            m.z *= m.step;
            // Quarantine a NaN/Inf mode so one bad mode can't poison the whole
            // bank permanently, and flush decayed modes to exactly zero so they
            // don't drift into subnormals (the classic audio-thread stall).
            const double zr = m.z.real();
            const double zi = m.z.imag();
            if (!std::isfinite(zr) || !std::isfinite(zi)
                || (std::abs(zr) < denorm_floor && std::abs(zi) < denorm_floor)) {
                m.z = {0.0, 0.0};
            }
        }
        if (!std::isfinite(sample)) sample = 0.0;
        // Hard limiter — never hand the DAC a sample outside [-1, 1].
        const float s = std::clamp(static_cast<float>(sample) * gain, -1.0F, 1.0F);
        for (ma_uint32 c = 0; c < channels; ++c)
            out[static_cast<std::size_t>(i) * channels + c] = s;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    AppState app;

    // Curved+ghost is the LMEAssembler default — the accuracy-
    // preferred path on open-boundary fixtures (32x8 polar disk
    // Leissa n=2: +0.75 % at defaults under the per-node-β
    // derivative fix `edaf0e1`; the flat path read +14.9 %). No
    // GUI-side opt-in needed; the Params{} defaults are right.
    //
    // The GUI ships with the 2nd-order SME (Rosolen-Millán-Arroyo
    // 2013) selected by default: it is the most ROBUST path across
    // fixture classes. On bending-dominated plates it ties 1st-order
    // LME (disk +0.73 % vs +0.75 %), and on membrane-dominated
    // curved shells it is far better behaved (free-free cylinder
    // +46 % vs LME's +274 % membrane locking; Scordelis-Lo ~1.002).
    // The user can untick "Use 2nd-order SME" on the LME tab to
    // fall back to the cheaper 1st-order LME.
    app.lme_params.use_second_order_sme = true;

    // Derive alpha and beta from the default eta (Steel preset).
    {
        const auto [a, b] = rayleigh_from_eta(static_cast<double>(app.eta));
        app.alpha = static_cast<float>(a);
        app.beta  = static_cast<float>(b);
    }

    // Mesh directory: prefer the build-time-configured CHLADNI_DATA_DIR
    // (absolute path baked in by CMake) so the binary works from any cwd;
    // otherwise fall back to a relative ./models path for unconfigured
    // dev builds.
    app.mesh_dir =
#ifdef CHLADNI_DATA_DIR
        fs::path{CHLADNI_DATA_DIR};
#else
        fs::path{"models"};
#endif
    scan_mesh_directory(app);

    const fs::path mesh_path =
        (argc > 1) ? fs::path{argv[1]} : (app.mesh_dir / "cylinder.obj");

    try {
        app.mesh = chladni::mesh::load_obj(mesh_path);
    } catch (const std::exception& e) {
        std::cerr << "strike_gui: failed to load '" << mesh_path.string()
                  << "': " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    app.current_mesh_name = mesh_path.filename().string();

    // Synchronise the dropdown index to the actually-loaded file so the
    // UI shows the right entry on first paint. Falls through to 0 if the
    // user passed an OBJ outside the dropdown's filtered set (either a
    // mesh too large for the size limit, or an explicit argv[1] path
    // outside @c models/).
    for (std::size_t i = 0; i < app.mesh_choice_filenames.size(); ++i) {
        if (app.mesh_choice_filenames[i] == app.current_mesh_name) {
            app.mesh_choice_idx = static_cast<int>(i);
            break;
        }
    }

    rebuild_geometry_from_mesh(app);
    std::cout << "strike_gui: loaded '" << app.current_mesh_name
              << "' (" << app.mesh.V.rows() << "v / "
              << app.mesh.F.rows() << "f); extruded "
              << app.V_rest_ext.rows() << "v / "
              << app.F_ext.rows() << "f\n";

    AudioState audio;
    app.audio = &audio;

    // ---- Solve modes at startup with default material — asynchronously.
    //
    // Kick the first eigensolve onto the worker thread (the same path
    // every later Recompute / mesh-swap uses) instead of running it
    // synchronously here. The synchronous version blocked *before*
    // polyscope::init(), so on a heavy mesh (e.g. bunny.obj, 14 k
    // vertices) the window never appeared until the solve finished —
    // the app looked frozen with nothing but the "loaded" line on the
    // terminal. Going async lets the window open immediately, show a
    // "computing..." status, and populate the audio bank when the
    // worker result is harvested by poll_recompute on a later frame.
    // The brief mode-less first frames are a fair trade for not
    // freezing the UI for tens of seconds (or longer) on large meshes.
    // poll_recompute only runs inside the render loop, i.e. after the
    // audio device below has opened, so apply_modes_result always sees
    // the real device sample rate when it builds the bank.
    kick_off_recompute(app);

    // ---- Polyscope.
    polyscope::init();
    polyscope::options::programName     = "chladni — strike_gui";
    polyscope::options::buildGui        = true;
    polyscope::options::buildDefaultGuiPanels = true;  // keep the structure list
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;  // suppress
    setup_polyscope_mesh(app);

    // ---- Drag-and-drop OBJ. Polyscope routes GLFW's drop event into
    // state::filesDroppedCallback; we just take the first .obj path
    // and feed it through the existing reload pipeline (auto-rescale,
    // re-register with Polyscope, recompute modes).
    polyscope::state::filesDroppedCallback =
        [&app](const std::vector<std::string>& paths) {
            for (const auto& p : paths) {
                fs::path fp{p};
                if (fp.extension() == ".obj" || fp.extension() == ".OBJ") {
                    if (load_mesh_from_path(app, fp)) {
                        // Selection no longer matches a bundled entry;
                        // a non-existent index is fine — the dropdown
                        // just shows whatever is at that slot, which
                        // the user can override on the next click.
                    }
                    return;
                }
            }
            std::cerr << "strike_gui: dropped files contained no .obj\n";
        };

    // ---- miniaudio.
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format  = ma_format_f32;
    cfg.playback.channels = 1;
    cfg.sampleRate       = 0;  // 0 = let miniaudio pick the device-native rate
    cfg.dataCallback     = audio_callback;
    cfg.pUserData        = &audio;

    ma_device device{};
    bool audio_ok = false;
    if (ma_device_init(nullptr, &cfg, &device) != MA_SUCCESS) {
        std::cerr << "strike_gui: ma_device_init FAILED — no audio.\n";
    } else if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "strike_gui: ma_device_start FAILED — no audio.\n";
        ma_device_uninit(&device);
    } else {
        audio_ok = true;
        // Use the device's actual rate for per-sample step factors.
        // Without this, hardcoding 44100 while the device opens at
        // 48000 produces an ~8.6 % pitch error (every mode plays flat).
        audio.sample_rate =
            static_cast<double>(device.playback.internalSampleRate);
        std::cout << "strike_gui: AUDIO STARTED ("
                  << device.playback.internalSampleRate << " Hz, "
                  << static_cast<int>(device.playback.internalChannels)
                  << " ch). Click on the mesh to strike.\n";
    }
    if (!audio_ok) {
        std::cout << "strike_gui: continuing without audio. The viz "
                  << "still works — strikes will produce visible deformation.\n";
    }

    // The startup eigensolve is now async (see kick_off_recompute above),
    // so at this point the mode bank is still empty — refresh_audio_slowdown
    // is a no-op until the worker's first result is harvested by
    // poll_recompute, which (running only inside the render loop, after this
    // device has opened) builds the bank at the real device sample rate.
    // We still set the high-cut default to the true Nyquist here so the UI
    // slider is right on the first frame.
    refresh_audio_slowdown(app);
    if (audio_ok) {
        const double nyquist_hz = 0.5 * audio.sample_rate;
        app.audio_high_cut_hz = std::min(
            app.spectrum_max_hz, static_cast<float>(nyquist_hz));
    }

    // ---- ImGui control panel + click handling.
    // Polyscope wraps the user callback in its "Command UI" window
    // (Polyscope source: "##Command UI"). Adding our own ImGui::Begin
    // here would create a *second* window, leaving the wrapper empty.
    // So all ImGui calls below go directly into the wrapper.
    polyscope::state::userCallback = [&app]() {
        // ---- Harvest worker results from any in-flight recompute.
        // Runs every frame; cheap (just a wait_for(0) on the future).
        poll_recompute(app);

        // ---- Mesh selection: dropdown over every .obj in mesh_dir.
        // Picking a different entry triggers a full reload (geometry,
        // Polyscope structure, eigensolve). Failures are non-fatal —
        // load_mesh_from_path leaves the previous mesh in place and
        // writes a one-line error into last_status.
        if (!app.mesh_choice_cstrs.empty()) {
            int prev_idx = app.mesh_choice_idx;
            if (ImGui::Combo("Mesh##selector", &app.mesh_choice_idx,
                             app.mesh_choice_cstrs.data(),
                             static_cast<int>(app.mesh_choice_cstrs.size()))) {
                const std::string& sel =
                    app.mesh_choice_filenames[
                        static_cast<std::size_t>(app.mesh_choice_idx)];
                if (!load_mesh_for_selection(app, sel)) {
                    // Revert the dropdown so the displayed selection
                    // matches what's actually loaded.
                    app.mesh_choice_idx = prev_idx;
                }
            }
            ImGui::TextDisabled("(drop an .obj on the window to load "
                                "external meshes)");
        }
        ImGui::Text("Mesh: %ld vertices, %ld triangles",
                    static_cast<long>(app.mesh.V.rows()),
                    static_cast<long>(app.mesh.F.rows()));

        // Procedural-mesh resolution. Only the sliders for the
        // currently active procedural geometry are shown; selecting a
        // different mesh from the dropdown swaps which subsection is
        // visible. Slider edits do NOT auto-regenerate (a 96x32
        // cylinder eigensolve is several seconds); a Regenerate
        // button below kicks off a full rebuild + recompute when the
        // user is done adjusting.
        if (!app.mesh_choice_cstrs.empty()
            && app.mesh_choice_idx >= 0
            && app.mesh_choice_idx < static_cast<int>(
                app.mesh_choice_filenames.size()))
        {
            const std::string& sel =
                app.mesh_choice_filenames[
                    static_cast<std::size_t>(app.mesh_choice_idx)];
            const bool is_proc = !sel.empty() && sel.front() == '*';
            if (is_proc) {
                ImGui::Indent();
                ImGui::TextUnformatted("Procedural resolution:");
                bool dirty = false;
                if (sel == kProcDisk) {
                    dirty |= ImGui::SliderInt(
                        "Disk n_azimuthal", &app.proc_disk_n_az, 3, 512);
                    dirty |= ImGui::SliderInt(
                        "Disk n_radial",    &app.proc_disk_n_rad, 1, 64);
                } else if (sel == kProcAnnulus) {
                    dirty |= ImGui::SliderInt(
                        "Annulus n_azimuthal",
                        &app.proc_annulus_n_az, 3, 512);
                    dirty |= ImGui::SliderInt(
                        "Annulus n_radial",
                        &app.proc_annulus_n_rad, 1, 64);
                    dirty |= ImGui::SliderFloat(
                        "Annulus b/a", &app.proc_annulus_b_over_a,
                        0.05F, 0.95F, "%.2f");
                } else if (sel == kProcCylinder) {
                    dirty |= ImGui::SliderInt(
                        "Cylinder n_around",
                        &app.proc_cylinder_n_around, 3, 256);
                    dirty |= ImGui::SliderInt(
                        "Cylinder n_along",
                        &app.proc_cylinder_n_along, 1, 128);
                } else if (sel == kProcIcosphere) {
                    dirty |= ImGui::SliderInt(
                        "Icosphere subdivisions",
                        &app.proc_icosphere_subdivisions, 0, 5);
                }
                // Quad-split scheme selector (disk / annulus /
                // cylinder; the icosphere is symmetric already). See
                // AppState::proc_quad_split / chladni::mesh::QuadSplit
                // for the chirality rationale + the Loop/LME caveat.
                if (sel != kProcIcosphere) {
                    static const char* kSplitNames[] = {
                        "consistent (legacy, chiral)",
                        "checkerboard (mirror-sym)",
                        "union-jack (full sym, best — SME only)",
                    };
                    int split_i = static_cast<int>(app.proc_quad_split);
                    if (ImGui::Combo("Triangulation", &split_i,
                                     kSplitNames, 3)) {
                        app.proc_quad_split =
                            static_cast<chladni::mesh::QuadSplit>(split_i);
                        dirty = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "How each grid quad is split into triangles.\n"
                            "consistent: one diagonal everywhere — the "
                            "mesh has a handedness that visibly\n  skews "
                            "the degenerate Chladni doublets (chirality).\n"
                            "checkerboard: parity-alternating diagonals — "
                            "exact mirror symmetry at the same\n  vertex "
                            "count, but the rim valence alternates 3/5 "
                            "and imprints its own rim artefacts.\n"
                            "union-jack: centre vertex + 4 triangles per "
                            "quad — full symmetry, uniform rims,\n  and "
                            "~12x better Leissa-disk accuracy (SME). "
                            "~2x triangle count.\n"
                            "Both symmetric schemes need an LME-family "
                            "assembler (1st- or 2nd-order,\n"
                            "the default): the Loop assembler rejects "
                            "their non-valence-4 rims.");
                    }
                }
                // Regenerate button — always enabled. Cheap if values
                // are unchanged (just re-runs the generator). Reuses
                // load_mesh_for_selection so the same install /
                // recompute path runs as a fresh selection.
                if (ImGui::Button("Regenerate procedural mesh")) {
                    load_mesh_for_selection(app, sel);
                }
                if (dirty) {
                    ImGui::SameLine();
                    ImGui::TextDisabled(
                        "(click Regenerate to apply)");
                }
                ImGui::Unindent();
            }
        }

        ImGui::Separator();
        ImGui::Text("Material");
        if (ImGui::Combo("Preset##material", &app.material_preset_idx,
                         kMaterialPresetNames,
                         IM_ARRAYSIZE(kMaterialPresetNames))) {
            if (app.material_preset_idx >= 0
                && app.material_preset_idx < kMaterialPresetCustom) {
                const auto& p = kMaterialPresets[app.material_preset_idx];
                app.E_GPa = p.E_GPa;
                app.nu    = p.nu;
                app.rho   = p.rho;
                app.eta   = p.eta;
                const auto [a, b] = rayleigh_from_eta(static_cast<double>(p.eta));
                app.alpha = static_cast<float>(a);
                app.beta  = static_cast<float>(b);
                // Damping is live; E/nu/rho still wait for Recompute.
                refresh_audio_damping(app);
            }
        }
        if (ImGui::SliderFloat("E (GPa)",      &app.E_GPa, 1.0F, 1000.0F, "%.1f",
                               ImGuiSliderFlags_Logarithmic)) {
            app.material_preset_idx = kMaterialPresetCustom;
        }
        if (ImGui::SliderFloat("nu",           &app.nu,   -0.49F,    0.49F, "%.3f")) {
            app.material_preset_idx = kMaterialPresetCustom;
        }
        if (ImGui::SliderFloat("rho (kg/m^3)", &app.rho,   100.0F, 20000.0F, "%.0f")) {
            app.material_preset_idx = kMaterialPresetCustom;
        }
        if (ImGui::SliderFloat("h (mm)", &app.h_mm,  0.05F, 10.0F, "%.2f")) {
            // Visual thickness tracks h immediately; modes still wait
            // for the explicit Recompute button (the eigensolve is the
            // expensive part).
            rebuild_render_extrusion(app);
        }

        ImGui::Separator();
        ImGui::SliderInt("# modes to solve", &app.n_modes_request, 5, 200);

        // Formulation tabs. One tab per ShellAssembler subclass. The
        // active tab drives `app.assembler` and exposes that subclass's
        // Params struct. Adding a future formulation (e.g.
        // WardetzkyAssembler) is purely additive — a new tab.
        //
        // After 9.8 the LME tab is mesh-shape-agnostic — the curved-
        // shell formulation (wPCA + Shepard PoU) handles both flat
        // and curved meshes from the same code path — so no
        // up-front flatness gate is needed any more.
        if (ImGui::BeginTabBar("##formulation_tabs",
                               ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Loop (Cirak-Ortiz 2000)")) {
                app.assembler = AssemblerKind::Loop;

                auto& p = app.loop_params;

                ImGui::Checkbox("Stam exact eval (irregular)", &p.use_stam);

                // n_passes is only meaningful for the L.3.4 multi-pass
                // drop path. Stam evaluates the irregular sub-tile
                // exactly at any depth, so extra passes don't change
                // its result — grey out the slider to make that explicit.
                ImGui::BeginDisabled(p.use_stam);
                ImGui::SliderInt("n_passes (Loop subdivision)",
                                 &p.n_passes, 1, 3);
                ImGui::EndDisabled();

                // Mass-lumping policy. Consistent (None) is the
                // shipped default since 2026-05-17 (late) — the
                // physically correct Galerkin formulation. RowSum
                // was the default 2026-05-13 → 2026-05-17 but
                // lumped mass systematically lowers eigenfrequencies
                // (the bias is most dramatic on closed shells: Loop
                // on icosphere k=2 sits 35 % below Wilkinson with
                // RowSum, 2 % above with None — pinned by the
                // [.experiment] Mass-lumping probe in
                // test_loop_sphere_projected.cpp). RowSum is kept
                // as a knob for cases where its high-m doublet
                // suppression is wanted, but the cost is a broad
                // frequency-lowering bias that's audible on most
                // meshes.
                ImGui::TextUnformatted("Mass lumping:");
                ImGui::SameLine();
                int lump_idx = (p.m_lump == chladni::shell::MassLumping::RowSum)
                                ? 1 : 0;
                if (ImGui::RadioButton("Consistent (None)", &lump_idx, 0)) {
                    p.m_lump = chladni::shell::MassLumping::None;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Row-sum lump", &lump_idx, 1)) {
                    p.m_lump = chladni::shell::MassLumping::RowSum;
                }

                // K / M reference-triangle quadrature. 1-pt centroid
                // is the original Cirak-Ortiz Sec 4.6 statics baseline
                // (exact for degree <= 1), 3-pt edge-mid is the
                // classical degree-2 rule, 7-pt Dunavant is the shipped
                // default (degree-5, over-integrates the degree-8
                // mass / degree-4 bending integrand with O(h^6)
                // residual). K and M can pick rules independently
                // for sensitivity studies.
                auto quad_radio = [](const char* label,
                                     chladni::shell::QuadratureRule& slot)
                {
                    using QR = chladni::shell::QuadratureRule;
                    int idx = (slot == QR::OnePointCentroid)   ? 0
                            : (slot == QR::ThreePointEdgeMid)  ? 1
                                                                : 2;
                    ImGui::TextUnformatted(label);
                    ImGui::SameLine();
                    if (ImGui::RadioButton((std::string{"1-pt##"}  + label).c_str(),
                                           &idx, 0)) slot = QR::OnePointCentroid;
                    ImGui::SameLine();
                    if (ImGui::RadioButton((std::string{"3-pt##"}  + label).c_str(),
                                           &idx, 1)) slot = QR::ThreePointEdgeMid;
                    ImGui::SameLine();
                    if (ImGui::RadioButton((std::string{"7-pt##"}  + label).c_str(),
                                           &idx, 2)) slot = QR::SevenPointDunavant;
                };
                quad_radio("K quadrature:", p.k_quad);
                quad_radio("M quadrature:", p.m_quad);

                ImGui::EndTabItem();
            }
            // LME tab — Arroyo-Ortiz 2006 meshfree LME. Curved-shell
            // mode (wPCA + Shepard PoU, Millán 2011) is enabled at app
            // startup, so this tab accepts both flat and curved meshes;
            // a flat plate runs the curved path as a degenerate case
            // (every chart aligns with the global xy plane) and a
            // curved mesh picks up the per-vertex tangent charts.
            // The single exposed knob is γ, the dimensionless aspect
            // ratio in β_a = γ / h_a² that sets basis support width.
            if (ImGui::BeginTabItem("LME (Arroyo-Ortiz)")) {
                app.assembler = AssemblerKind::LME;

                auto& p = app.lme_params;

                // 2nd-order SME (Rosolen-Millán-Arroyo 2013) toggle.
                // ON by default in the GUI (set at startup) — it is the
                // most robust path: ties 1st-order LME on bending-
                // dominated plates (disk +0.73 % vs +0.75 %) and avoids
                // LME's membrane locking on curved shells (cylinder
                // +46 % vs +274 %). Untick to fall
                // back to the cheaper 1st-order LME. When on, the
                // per-Gauss-point in-chart
                // basis dispatches to evaluate_sme_basis_grad_and_hess
                // and the γ slider below has no effect (SME's locality
                // emerges from per-node μ★ being positive-definite,
                // not from a Gaussian prior). Reference accuracy gate:
                // 32x8 polar disk Leissa (n=2) — both sub-1 %
                // (LME +0.75 %, SME +0.73 %, gated at 2 %).
                ImGui::Checkbox("Use 2nd-order SME (Rosolen 2013)",
                                &p.use_second_order_sme);

                // γ: shipped default 0.8 = Millán 2011 Table I thin-shell
                // γ_LME (the "1.6" in older notes was that paper's Fig-4 2D
                // non-shell demo). Range [0.5, 4.0]. γ → 0 widens / smooths
                // the basis (Strang-Fix friendly), γ → ∞ sharpens to
                // near-Delaunay-hat. NOTE: γ=0.8 is disk-optimal but locks
                // 1st-order LME on free-free CURVED membrane shells (the
                // bundled cylinder); raise γ (~2.5) there, or use SME/Loop.
                // Greyed out when SME is on — SME ignores γ and uses the
                // slack-matrix scalars α (sme_alpha) instead.
                if (p.use_second_order_sme) ImGui::BeginDisabled();
                auto gamma_f = static_cast<float>(p.gamma);
                if (ImGui::SliderFloat(
                        "γ (LME basis aspect ratio)",
                        &gamma_f, 0.5F, 4.0F, "%.2f"))
                {
                    p.gamma = static_cast<double>(gamma_f);
                }
                if (p.use_second_order_sme) ImGui::EndDisabled();

                // α: SME interior-slack scalar in d_a = (α/4) h² I.
                // Default 2.0 (the paper's α=2; faithful §3.2.2 recipe).
                // Only meaningful on the SME path.
                if (!p.use_second_order_sme) ImGui::BeginDisabled();
                auto alpha_f = static_cast<float>(p.sme_alpha);
                if (ImGui::SliderFloat(
                        "α (SME interior slack)",
                        &alpha_f, 1.5F, 16.0F, "%.2f"))
                {
                    p.sme_alpha = static_cast<double>(alpha_f);
                }
                if (!p.use_second_order_sme) ImGui::EndDisabled();

                ImGui::TextDisabled(
                    "Meshfree max-entropy (Arroyo-Ortiz 2006 /\n"
                    "Rosolen 2013). 2nd-order SME is the default —\n"
                    "more robust than 1st-order LME on curved shells.\n"
                    "Slower than Loop; prefer Loop for interactive\n"
                    "work on large meshes. Accuracy figures: see\n"
                    "the Verification section of the README.");

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Legacy (CST+IBM)")) {
                app.assembler = AssemblerKind::Legacy;
                ImGui::TextDisabled(
                    "Pre-2026 CST membrane + Wardetzky-IBM bending.\n"
                    "Over-stiff vs analytic fixtures — kept as a\n"
                    "fallback when Loop's boundary handling rejects a\n"
                    "mesh (e.g. valence-2 strip corners).");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        // Async dispatch: the button kicks off a worker thread; we
        // grey-out and relabel it while the worker is in flight so
        // the user can't pile up redundant recomputes. The window
        // close button stays live because the UI thread isn't
        // blocked on the eigensolve.
        if (app.compute_in_flight) {
            ImGui::BeginDisabled();
            ImGui::Button("Computing... (close window OK)");
            ImGui::EndDisabled();
        } else {
            if (ImGui::Button("Recompute modes")) {
                rebuild_render_extrusion(app);
                kick_off_recompute(app);
            }
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(app.last_status.c_str());

        if (app.modes_valid && app.modes.omegas.size() > 0) {
            const double two_pi = 2.0 * std::numbers::pi_v<double>;
            ImGui::Text("Spectrum: %.1f Hz — %.1f Hz",
                        static_cast<double>(app.spectrum_min_hz),
                        static_cast<double>(app.spectrum_max_hz));

            const double low  = static_cast<double>(app.audio_low_cut_hz);
            const double high = static_cast<double>(app.audio_high_cut_hz);
            int n_in_band = 0;
            for (Eigen::Index k = 0; k < app.modes.omegas.size(); ++k) {
                const double f_k = app.modes.omegas(k) / two_pi;
                if (f_k >= low && f_k <= high) ++n_in_band;
            }
            ImGui::Text("Modes inside band: %d / %d",
                        n_in_band, static_cast<int>(app.modes.omegas.size()));
        }

        ImGui::Separator();
        ImGui::Text("Damping");
        // ImGui derives a logarithmic "zero epsilon" = 0.1^(format decimal
        // precision) and, when a logarithmic slider's WHOLE [v_min, v_max]
        // range sits below that epsilon, it fudges both endpoints up to the
        // epsilon and pins the grip to the far left for every in-range value
        // — the bar then has only two reachable states and never moves
        // (imgui_widgets.cpp ScaleRatioFromValueT). TWO traps bite here:
        //   (1) any scientific "%.Ne" format makes ImParseFormatPrecision
        //       return -1 ("maximum precision"), so epsilon = 0.1^-1 = 10 —
        //       N is ignored. eta (~1e-4) then sits entirely below epsilon
        //       and the slider is dead. So eta MUST use a fixed "%.Nf"
        //       format: "%.6f" gives epsilon 1e-6, below its 1e-5 v_min.
        //   (2) beta (~1e-9 s) is far below any sane fixed-format epsilon,
        //       so it is edited through an O(1) nanosecond proxy below.
        // alpha (1e-3 .. 200, "%.3f") already clears its 1e-3 epsilon.
        if (ImGui::SliderFloat("eta (loss factor)", &app.eta,
                               1.0e-5F, 1.0e-1F, "%.6f",
                               ImGuiSliderFlags_Logarithmic)) {
            const auto [a, b] = rayleigh_from_eta(static_cast<double>(app.eta));
            app.alpha = static_cast<float>(a);
            app.beta  = static_cast<float>(b);
            app.material_preset_idx = kMaterialPresetCustom;
            refresh_audio_damping(app);
        }
        ImGui::TextDisabled("(Rayleigh d_i = (alpha + beta omega_i^2) / 2; "
                            "anchored at 200 Hz and 5 kHz. eta drives both; "
                            "dragging alpha or beta back-solves eta.)");
        // Positive lower bound: a logarithmic slider has no finite image of
        // zero (the ImGui grip behaves erratically across the lower decade if
        // v_min == 0). 1e-3 is negligible mass-proportional damping — the
        // useful range is eta in [1e-5, 1e-1] -> alpha in ~[0.012, 121].
        if (ImGui::SliderFloat("alpha (1/s)", &app.alpha,
                               1.0e-3F, 200.0F, "%.3f", ImGuiSliderFlags_Logarithmic)) {
            // Back-solve the master eta dial so it stays consistent with the
            // axis just dragged; beta is left as set (second Rayleigh DOF).
            app.eta = static_cast<float>(eta_from_alpha(static_cast<double>(app.alpha)));
            app.material_preset_idx = kMaterialPresetCustom;
            refresh_audio_damping(app);
        }
        // beta is ~1e-9 s — far below any short-format logarithmic epsilon
        // (see the eta note). Edit it as nanoseconds (O(0.1..1e4)) so the
        // grip tracks; [1e-10, 1e-5] s == [0.1, 1e4] ns.
        float beta_ns = app.beta * 1.0e9F;
        if (ImGui::SliderFloat("beta (ns)",   &beta_ns,
                               1.0e-1F, 1.0e4F, "%.2f", ImGuiSliderFlags_Logarithmic)) {
            app.beta = beta_ns * 1.0e-9F;
            // Back-solve the master eta dial (see the alpha branch above).
            app.eta = static_cast<float>(eta_from_beta(static_cast<double>(app.beta)));
            app.material_preset_idx = kMaterialPresetCustom;
            refresh_audio_damping(app);
        }

        ImGui::Separator();
        ImGui::Text("Excitation");
        // Mutually exclusive Hammer / Single-mode tabs. Hammer is the
        // click-on-mesh impulse path; Single-mode is the slider-driven
        // steady ring (textbook Chladni demo). Switching tabs silences
        // the audio bank so leftover sound from the previous mode can't
        // bleed through; switching INTO single-mode also immediately
        // re-rings the currently selected k/amp so the user sees a
        // pattern without wiggling a slider first.
        const auto prev_mode = app.excitation_mode;

        if (ImGui::BeginTabBar("##excitation_tabs",
                               ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Hammer##exc")) {
                app.excitation_mode = AppState::ExcitationMode::Hammer;

                if (ImGui::Combo("Preset", &app.hammer_preset_idx,
                                 kHammerPresetNames,
                                 IM_ARRAYSIZE(kHammerPresetNames))) {
                    // Selecting any concrete preset writes its values
                    // into the sliders; "Custom" leaves them as-is.
                    if (app.hammer_preset_idx >= 0
                        && app.hammer_preset_idx < kHammerPresetCustom) {
                        app.hammer_force =
                            kHammerPresets[app.hammer_preset_idx].force;
                        app.hammer_cutoff_hz =
                            kHammerPresets[app.hammer_preset_idx].cutoff_hz;
                    }
                }
                if (ImGui::SliderFloat("Force", &app.hammer_force,
                                       0.0F, 2.0F, "%.2f")) {
                    app.hammer_preset_idx = kHammerPresetCustom;
                }
                if (ImGui::SliderFloat("Cutoff Hz", &app.hammer_cutoff_hz,
                                       50.0F, 50000.0F, "%.0f",
                                       ImGuiSliderFlags_Logarithmic)) {
                    app.hammer_preset_idx = kHammerPresetCustom;
                }
                ImGui::SliderFloat("Strike intensity",
                                   &app.strike_intensity,
                                   0.1F, 1000.0F, "%.1f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Single mode##exc")) {
                app.excitation_mode = AppState::ExcitationMode::SingleMode;

                if (app.modes_valid && app.modes.omegas.size() > 0) {
                    const int n_modes_avail =
                        static_cast<int>(app.modes.omegas.size());
                    // Clamp the spinner if the spectrum shrunk (e.g.
                    // after a recompute with fewer modes or a coarser
                    // mesh).
                    if (app.single_mode_k_one_based > n_modes_avail) {
                        app.single_mode_k_one_based = n_modes_avail;
                    }
                    if (app.single_mode_k_one_based < 1) {
                        app.single_mode_k_one_based = 1;
                    }
                    // Single-mode is always "live" within its panel:
                    // any slider edit immediately re-rings the selected
                    // mode. Sweeping Mode k walks through the Chladni-
                    // pattern catalogue (set alpha, beta near 0 in the
                    // material panel for stationary nodal lines).
                    const bool k_changed = ImGui::SliderInt(
                        "Mode k", &app.single_mode_k_one_based,
                        1, n_modes_avail);
                    constexpr double tau_ui =
                        2.0 * std::numbers::pi_v<double>;
                    const double f_k =
                        app.modes.omegas(app.single_mode_k_one_based - 1)
                        / tau_ui;
                    ImGui::Text("Frequency: %.2f Hz", f_k);
                    const bool amp_changed = ImGui::SliderFloat(
                        "Amplitude##single_mode",
                        &app.single_mode_amplitude,
                        0.001F, 10.0F, "%.3f",
                        ImGuiSliderFlags_Logarithmic);
                    // Sound checkbox sits below the sliders so the
                    // slider edits (the main interaction) are the
                    // visual focus and the audio toggle is the "extra
                    // knob". Default off — the textbook Chladni use
                    // case is visual, not aural.
                    const bool sound_toggled = ImGui::Checkbox(
                        "Sound##single_mode", &app.single_mode_sound);
                    ImGui::TextUnformatted(
                        "(set alpha, beta near 0 for a stationary "
                        "pattern)");

                    if (sound_toggled || k_changed || amp_changed) {
                        ring_single_mode(
                            app, app.single_mode_k_one_based,
                            static_cast<double>(
                                app.single_mode_amplitude));
                    }
                } else {
                    ImGui::TextDisabled("(modes not yet computed)");
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        // Tab-switch side effects: silence the audio bank and (when
        // entering single-mode with valid modes) re-ring the selected k.
        if (app.excitation_mode != prev_mode) {
            if (app.audio != nullptr) {
                std::lock_guard<std::mutex> lk(app.audio->mutex);
                for (auto& m : app.audio->modes) {
                    m.z = std::complex<double>(0.0, 0.0);
                }
            }
            for (auto& z : app.viz_z) z = std::complex<double>(0.0, 0.0);
            if (app.excitation_mode == AppState::ExcitationMode::SingleMode
                && app.modes_valid && app.modes.omegas.size() > 0)
            {
                ring_single_mode(
                    app, app.single_mode_k_one_based,
                    static_cast<double>(app.single_mode_amplitude));
            }
        }

        ImGui::Separator();
        ImGui::Text("Audio band (auto-fit to the computed spectrum)");
        ImGui::SliderFloat("Low cut (Hz)",  &app.audio_low_cut_hz,
                           app.spectrum_min_hz, app.spectrum_max_hz,
                           "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("High cut (Hz)", &app.audio_high_cut_hz,
                           app.spectrum_min_hz, app.spectrum_max_hz,
                           "%.0f", ImGuiSliderFlags_Logarithmic);
        if (app.audio_low_cut_hz > app.audio_high_cut_hz) {
            // Keep low <= high so the band stays a band.
            app.audio_low_cut_hz = app.audio_high_cut_hz;
        }

        ImGui::Separator();
        ImGui::Text("Time dilation (couples audio pitch and viz speed)");
        ImGui::Checkbox("Enable viz",  &app.viz_enabled);
        if (ImGui::SliderFloat("Slowdown", &app.viz_slowdown, 1.0F, 10000.0F, "%.1fx",
                               ImGuiSliderFlags_Logarithmic)) {
            refresh_audio_slowdown(app);
        }
        ImGui::SliderFloat("Amplification", &app.viz_amplification, 1.0e-4F, 10.0F, "%.4f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::SliderFloat("Wall thickness x", &app.viz_thickness_scale,
                               1.0F, 30.0F, "%.1f")) {
            rebuild_render_extrusion(app);
        }

        if (ImGui::Checkbox("Chladni overlay", &app.viz_chladni_enabled)) {
            if (app.ps_chladni_q != nullptr) {
                app.ps_chladni_q->setEnabled(app.viz_chladni_enabled);
            }
        }
        ImGui::SliderFloat("Isolines (1 = nodal set only)",
                           &app.viz_isoline_count, 1.0F, 64.0F, "%.0f");
        ImGui::SliderFloat("Isoline thickness",
                           &app.viz_isoline_thickness, 0.0F, 1.0F, "%.2f");
        ImGui::SliderFloat("Isoline darkness",
                           &app.viz_isoline_darkness, 0.0F, 1.0F, "%.2f");
        // Loop-subdivide the *render* mesh by this many passes to
        // smooth the nodal-line overlay (Polyscope interpolates the
        // scalar linearly across each triangle; denser triangles =
        // smoother lines). FEM solve is unaffected. Cost: 4^N more
        // viz vertices.
        if (ImGui::SliderInt("Render subdivision (viz only)",
                             &app.viz_subdivision_passes, 0, 2))
        {
            rebuild_geometry_from_mesh(app);
            setup_polyscope_mesh(app);
        }
        // Threshold below which the rendered scalar is forced
        // off-contour (no isoline drawn). Targets the fractal noise
        // pattern at the centre of high-m modes where amplitude ~ r^m
        // falls below truncation noise. The lower bound is a small
        // positive value (effectively off): a logarithmic slider has no
        // finite image of zero, so v_min = 0 makes the grip erratic
        // across the low decade (same pitfall as the alpha slider).
        ImGui::SliderFloat("Noise-floor suppression",
                           &app.viz_chladni_noise_floor,
                           1.0e-5F, 1.0e-1F, "%.5f",
                           ImGuiSliderFlags_Logarithmic);

        // ---- Mode gallery: save the lowest N mode shapes as PNGs ----
        // Each click iterates through modes 0 .. N-1 across successive
        // frames (one mode per frame), freezing viz_z to imaginary unit
        // amplitude for the active mode and zero elsewhere, then calling
        // polyscope::screenshot(). Output goes to a timestamped
        // subdirectory of screenshots/gallery/ so successive runs don't
        // overwrite each other.
        ImGui::Separator();
        ImGui::Text("Mode gallery");
        ImGui::SliderInt("Gallery count", &app.gallery_count, 1, 50);
        ImGui::SliderFloat("Gallery amplitude", &app.gallery_amp,
                           1.0e-5F, 1.0e-1F, "%.5f",
                           ImGuiSliderFlags_Logarithmic);
        if (app.gallery_active) {
            ImGui::Text("Saving mode %d / %d...",
                        app.gallery_index + 1, app.gallery_count);
        } else if (ImGui::Button("Save mode gallery (PNG per mode)")) {
            if (app.modes_valid && app.modes.omegas.size() > 0) {
                using std::chrono::system_clock;
                const auto now_t = system_clock::to_time_t(
                    system_clock::now());
                char stamp[64];
                std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S",
                              std::localtime(&now_t));
                std::filesystem::path dir =
                    std::filesystem::path("screenshots") / "gallery" /
                    (std::string{stamp} + "_" +
                     app.compute_params.assembly_label);
                // sanitize spaces in label for path safety
                std::string s = dir.string();
                std::replace(s.begin(), s.end(), ' ', '_');
                std::replace(s.begin(), s.end(), '(', '_');
                std::replace(s.begin(), s.end(), ')', '_');
                std::replace(s.begin(), s.end(), '+', '_');
                app.gallery_dir = s;
                std::error_code ec;
                std::filesystem::create_directories(app.gallery_dir, ec);
                app.gallery_index  = 0;
                app.gallery_active = true;
                std::cout << "gallery: saving "
                          << app.gallery_count << " modes to "
                          << app.gallery_dir << '\n';
            }
        }

        ImGui::Separator();
        const bool hammer_active =
            app.excitation_mode == AppState::ExcitationMode::Hammer;
        ImGui::BeginDisabled(!hammer_active);
        ImGui::Text("Click on the mesh to strike. Strikes: %d", app.strike_count);
        if (ImGui::Button("Strike vertex 0 (debug)")) {
            apply_strike(app, 0);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Silence audio bank")) {
            // Hard reset of the resonator state: zeroes every mode's
            // current oscillator coordinate without rebuilding the bank.
            // Useful when something weird has happened to the bank
            // (NaN, phantom DC, etc.) and you want a clean slate
            // without paying for a full Recompute.
            if (app.audio != nullptr) {
                std::lock_guard<std::mutex> lk(app.audio->mutex);
                for (auto& m : app.audio->modes) {
                    m.z = std::complex<double>(0.0, 0.0);
                }
            }
            for (auto& z : app.viz_z) z = std::complex<double>(0.0, 0.0);
        }

        // ---- Live audio-bank diagnostic. Reads the bank with
        // try_lock so we never block the audio thread. Shows how many
        // modes still have non-trivial oscillator state and the peak
        // |z| — if this is 0 with a non-zero strike count, the bank
        // is effectively silent (heavy damping, nuked by H_band, NaN
        // in step, etc.).
        if (app.audio != nullptr) {
            int    n_total   = 0;
            int    n_ringing = 0;
            double max_abs_z = 0.0;
            double max_abs_step = 0.0;
            double min_abs_step = 1.0;
            bool   has_nan   = false;
            std::unique_lock<std::mutex> lk(app.audio->mutex,
                                            std::try_to_lock);
            if (lk.owns_lock()) {
                n_total = static_cast<int>(app.audio->modes.size());
                for (const auto& m : app.audio->modes) {
                    const double az = std::abs(m.z);
                    const double as = std::abs(m.step);
                    if (std::isnan(az) || std::isnan(as)) has_nan = true;
                    if (az > 1.0e-9) ++n_ringing;
                    max_abs_z    = std::max(max_abs_z, az);
                    max_abs_step = std::max(max_abs_step, as);
                    min_abs_step = std::min(min_abs_step, as);
                }
                ImGui::Text("Audio bank: %d/%d ringing, max |z|=%.3e",
                            n_ringing, n_total, max_abs_z);
                ImGui::Text("|step| in [%.6f, %.6f]%s",
                            min_abs_step, max_abs_step,
                            has_nan ? "  *** NaN DETECTED ***" : "");
            } else {
                ImGui::TextDisabled("Audio bank: (busy)");
            }
        }

        // Mode-gallery driver. One step per frame: set viz_z to a
        // pure-imaginary unit at the active mode (so Im() reads as the
        // mode's full amplitude immediately, no dynamic build-up), call
        // update_viz_mesh with dt=0 to push positions + Chladni scalar
        // without integrating the oscillator state, then request a
        // screenshot via polyscope. Polyscope queues the screenshot for
        // the next render pass so we get the just-updated frame.
        if (app.gallery_active) {
            const int k = app.gallery_index;
            const int n_modes_avail =
                static_cast<int>(app.modes.omegas.size());
            if (k < n_modes_avail
                && app.viz_z.size() == static_cast<std::size_t>(n_modes_avail))
            {
                for (auto& z : app.viz_z) {
                    z = std::complex<double>(0.0, 0.0);
                }
                app.viz_z[static_cast<std::size_t>(k)] =
                    std::complex<double>(
                        0.0, static_cast<double>(app.gallery_amp));
                update_viz_mesh(app, /*frame_dt=*/0.0);

                char fname[64];
                std::snprintf(fname, sizeof(fname),
                              "mode_%03d.png", k);
                const std::string path =
                    (std::filesystem::path(app.gallery_dir) / fname).string();
                polyscope::screenshot(path, /*transparentBG=*/false);
                std::cout << "  saved " << path << '\n';
            }
            ++app.gallery_index;
            if (app.gallery_index >= app.gallery_count
                || app.gallery_index >= n_modes_avail)
            {
                app.gallery_active = false;
                // Clear viz so we return to normal state cleanly.
                for (auto& z : app.viz_z) {
                    z = std::complex<double>(0.0, 0.0);
                }
                std::cout << "gallery: done.\n";
            }
            // Skip the regular dynamics step this frame.
            return;
        }

        // Slow-motion deformation update each frame.
        const double now = ImGui::GetTime();
        const double frame_dt = (app.viz_last_time > 0.0)
            ? std::min(now - app.viz_last_time, 0.1)  // cap dt at 100 ms
            : 1.0 / 60.0;
        app.viz_last_time = now;
        update_viz_mesh(app, frame_dt);

        // Click handling: take Polyscope's 3D hit position and strike the
        // mesh vertex nearest to it. This sidesteps version-specific
        // localIndex layouts (Polyscope's element-ordering convention has
        // shifted between releases).
        ImGuiIO& io = ImGui::GetIO();
        // Click-to-strike fires only in Hammer excitation mode. In
        // Single-mode the mesh is non-interactive — the single_mode_*
        // sliders are the only way to drive a mode.
        const bool can_click_strike =
            app.excitation_mode == AppState::ExcitationMode::Hammer;
        if (can_click_strike
            && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
            && !io.WantCaptureMouse)
        {
            const glm::vec2 mouse_pos{io.MousePos.x, io.MousePos.y};
            const polyscope::PickResult pick =
                polyscope::pickAtScreenCoords(mouse_pos);
            if (pick.isHit && pick.structure != nullptr
                && pick.structure->name == "shell") {
                const Eigen::Vector3d hit{
                    static_cast<double>(pick.position.x),
                    static_cast<double>(pick.position.y),
                    static_cast<double>(pick.position.z)};
                Eigen::Index nearest = -1;
                double best = std::numeric_limits<double>::max();
                for (Eigen::Index i = 0; i < app.V_rest.rows(); ++i) {
                    const double d2 =
                        (app.V_rest.row(i).transpose() - hit).squaredNorm();
                    if (d2 < best) {
                        best    = d2;
                        nearest = i;
                    }
                }
                if (nearest >= 0) {
                    std::cout << "click hit mesh, nearest vertex=" << nearest
                              << " dist=" << std::sqrt(best) << "\n";
                    apply_strike(app, nearest);
                }
            }
        }
    };

    polyscope::show();

    if (ma_device_get_state(&device) != ma_device_state_uninitialized) {
        ma_device_uninit(&device);
    }
    return EXIT_SUCCESS;
}
