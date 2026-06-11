# ---------------------------------------------------------------------------
# chladni applications
# ---------------------------------------------------------------------------
add_subdirectory(synth_demo)
add_subdirectory(chladni_figures)

if(CHLADNI_BUILD_GUI)
    add_subdirectory(strike_gui)
endif()
