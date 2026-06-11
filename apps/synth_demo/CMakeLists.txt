# ---------------------------------------------------------------------------
# synth_demo — listen to a drum-shell-ish spectrum
#
# Generates a WAV file from a hand-picked material + geometry so we can
# verify the synth + WAV path with our ears. Loads cylinder.obj, scales it
# to drum-shell dimensions, solves the 30 lowest free-free shell modes via
# compute_shell_modes, applies Rayleigh damping + a hammer impulse, and
# writes the resonator-bank impulse response to a WAV.
# ---------------------------------------------------------------------------
add_executable(synth_demo main.cpp)
target_link_libraries(synth_demo
    PRIVATE
        chladni::chladni
        chladni_warnings
)
# Make the source-tree models/ discoverable so the demo can find
# cylinder.obj when run with no arguments.
target_compile_definitions(synth_demo PRIVATE
    CHLADNI_DATA_DIR="${CMAKE_SOURCE_DIR}/models"
)
