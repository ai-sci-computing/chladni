# ---------------------------------------------------------------------------
# strike_gui — interactive shell-strike GUI on top of Polyscope.
#
# Phase 3 entry point: load a mesh, pick material/hammer, click to
# strike, hear the sound. Not built when CHLADNI_BUILD_GUI=OFF.
# ---------------------------------------------------------------------------
add_executable(strike_gui main.cpp)

target_link_libraries(strike_gui
    PRIVATE
        chladni::chladni
        chladni_warnings
        polyscope
        miniaudio_header
)

# macOS audio frameworks for miniaudio's CoreAudio backend.
if(APPLE)
    target_link_libraries(strike_gui PRIVATE
        "-framework CoreAudio"
        "-framework CoreFoundation"
        "-framework AudioToolbox"
        "-framework AudioUnit")
endif()

target_compile_definitions(strike_gui PRIVATE
    CHLADNI_DATA_DIR="${CMAKE_SOURCE_DIR}/models"
)
