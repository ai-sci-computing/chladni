# ---------------------------------------------------------------------------
# chladni — modal sound synthesis from surface meshes
#
# Top-level build script. Dependencies are fetched from source via
# FetchContent so a fresh checkout requires only CMake (>=3.25) and a
# C++20-capable compiler. No system packages are needed.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.25)

project(chladni
    VERSION     0.1.0
    DESCRIPTION "Modal sound synthesis from surface meshes"
    LANGUAGES   CXX
)

# --- Language standard --------------------------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- User-facing options ------------------------------------------------
option(CHLADNI_BUILD_TESTS "Build the Catch2 test suite"            ON)
option(CHLADNI_BUILD_APPS  "Build the CLI and GUI applications"     ON)
option(CHLADNI_BUILD_GUI   "Build the interactive strike_gui (pulls Polyscope)" ON)
option(CHLADNI_BUILD_DOCS  "Generate Doxygen API documentation"     OFF)

# --- Common warning interface -------------------------------------------
add_library(chladni_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(chladni_warnings INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wconversion -Wsign-conversion
        -Wdouble-promotion -Wformat=2)
endif()

# --- Dependencies via FetchContent --------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Eigen — dense and sparse linear algebra.
# SYSTEM marks Eigen's include directories as system headers; this
# silences strict-warning chatter (sign-conversion etc.) emitted from
# inside its templates while preserving those warnings for our own code.
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3147391d946bb4b6c68edd901f2add6ac1f31f8c  # 3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# Spectra — header-only sparse generalised eigensolver built on Eigen.
FetchContent_Declare(spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        bdd707b9a872bb17622696ed3cca2a27b743a5b7  # v1.0.1
    GIT_SHALLOW    TRUE
    SYSTEM
)
FetchContent_MakeAvailable(spectra)

# tinyobjloader — single-header Wavefront OBJ reader.
# Used by chladni::mesh::load_obj. Linked PRIVATE to chladni so its
# header does not leak to consumers.
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        2945a967c5303b2c8c14174117c45f3302591150  # v2.0.0rc13
    GIT_SHALLOW    TRUE
    SYSTEM
)
set(TINYOBJLOADER_USE_DOUBLE     OFF CACHE BOOL "" FORCE)
set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE BOOL "" FORCE)
set(TINYOBJLOADER_BUILD_OBJ_STICHER OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinyobjloader)

# Polyscope — 3D viewer + ImGui integration. Heavyweight: pulls GLFW,
# glm, glad, Dear ImGui, json. Gated on CHLADNI_BUILD_GUI so users who
# only need the library and CLI demos can skip the cost.
if(CHLADNI_BUILD_GUI)
    FetchContent_Declare(polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG        0c3dd68b9851417e6b2b976d347adc3250026122  # v2.6.1
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(polyscope)

    # miniaudio — single-header realtime audio output. We expose its
    # source directory as an INTERFACE include path; the consumer
    # (strike_gui) defines MINIAUDIO_IMPLEMENTATION in exactly one TU.
    FetchContent_Declare(miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG        4a5b74bef029b3592c54b6048650ee5f972c1a48  # 0.11.21
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(miniaudio)
    add_library(miniaudio_header INTERFACE)
    target_include_directories(miniaudio_header SYSTEM INTERFACE
        ${miniaudio_SOURCE_DIR})
endif()

# --- Core library -------------------------------------------------------
add_library(chladni STATIC
    src/analytical/beam.cpp
    src/analytical/plate.cpp
    src/analytical/shell.cpp
    src/detail/bessel.cpp
    src/mesh.cpp
    src/shell.cpp
    src/shell/assembler.cpp
    src/shell/lme.cpp
    src/shell/loop.cpp
    src/shell/loop_stam.cpp
    src/synth.cpp
    src/wav.cpp
)
add_library(chladni::chladni ALIAS chladni)

target_include_directories(chladni PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(chladni
    PUBLIC  Eigen3::Eigen
    PRIVATE chladni_warnings tinyobjloader Spectra)

# --- Tests --------------------------------------------------------------
if(CHLADNI_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        b5373dadca40b7edc8570cf9470b9b1cb1934d40  # v3.5.4
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    add_subdirectory(tests)
endif()

# --- Applications -------------------------------------------------------
if(CHLADNI_BUILD_APPS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt")
    add_subdirectory(apps)
endif()

# --- Doxygen documentation ---------------------------------------------
if(CHLADNI_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_GENERATE_HTML  YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTRACT_ALL    YES)
    # Force a bright, light theme (don't follow the OS dark mode).
    set(DOXYGEN_HTML_COLORSTYLE       LIGHT)
    set(DOXYGEN_HTML_COLORSTYLE_HUE   209)
    set(DOXYGEN_HTML_COLORSTYLE_SAT   120)
    set(DOXYGEN_HTML_COLORSTYLE_GAMMA 95)
    # Render @f$ … @f$ math with MathJax (client-side) instead of generating
    # formula PNGs via a local LaTeX toolchain — avoids broken-image formulas.
    set(DOXYGEN_USE_MATHJAX     YES)
    set(DOXYGEN_MATHJAX_VERSION MathJax_3)
    set(DOXYGEN_MATHJAX_RELPATH https://cdn.jsdelivr.net/npm/mathjax@3)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    # Resolve @cite tags against the bibliography (doxygen_add_docs generates
    # its own Doxyfile, so the standalone ./Doxyfile's CITE_BIB_FILES is unused).
    set(DOXYGEN_CITE_BIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/papers/references.bib)
    # Resolve GitHub-style markdown anchors ([..](#section-title)) in the
    # README so the intra-page links survive the Doxygen rendering.
    set(DOXYGEN_MARKDOWN_ID_STYLE GITHUB)
    # Use the README as the docs landing page (otherwise index.html is empty
    # and the API only shows under the Namespaces/Classes/Files tabs).
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR})  # resolves images/chladni.png
    # Cap the hero image size on the generated main page.
    set(DOXYGEN_HTML_EXTRA_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-extra.css)
    doxygen_add_docs(docs
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        COMMENT "Generating Doxygen API documentation"
    )
endif()
