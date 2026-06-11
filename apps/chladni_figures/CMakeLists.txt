# ---------------------------------------------------------------------------
# chladni_figures — render a 6x6 contact sheet of Chladni nodal patterns
# (the project hero image). Graphics-free: solves modes and rasterizes the
# nodal sets to a PPM, no Polyscope / GUI dependency.
# ---------------------------------------------------------------------------
add_executable(chladni_figures main.cpp)
target_link_libraries(chladni_figures
    PRIVATE
        chladni::chladni
        chladni_warnings
)
