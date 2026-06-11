# ---------------------------------------------------------------------------
# Test suite — Catch2 v3
#
# Catch2 is fetched with SYSTEM in the top-level CMakeLists.txt so that
# its own warnings (notably -Wnon-virtual-dtor on BinaryExpr templates)
# do not leak into our build.
# ---------------------------------------------------------------------------
include(Catch)

add_executable(chladni_tests
    analytical/test_beam_free_free.cpp
    analytical/test_plate_annular.cpp
    analytical/test_plate_free_edge_circular.cpp
    analytical/test_plate_simply_supported.cpp
    analytical/test_shell_cylindrical.cpp
    analytical/test_shell_spherical.cpp
    detail/test_bessel.cpp
    shell/test_energy.cpp
    shell/test_geometry.cpp
    shell/test_loop_basis.cpp
    shell/test_loop_boundary.cpp
    shell/test_loop_canonical.cpp
    shell/test_loop_element_K.cpp
    shell/test_loop_evaluate.cpp
    shell/test_loop_gather_stam_dofs.cpp
    shell/test_loop_global_K.cpp
    # shell/test_loop_multipoint_quad.cpp  # archived: now-stale experiment
                                            #   API (per-point K) no longer
                                            #   exists; investigation notes
                                            #   preserved in the source file.
    shell/test_loop_sphere_projected.cpp
    shell/test_loop_all_bundled_meshes.cpp
    shell/test_loop_assembler_pinning.cpp
    shell/test_lme_annulus.cpp
    shell/test_lme_assembler.cpp
    shell/test_lme_basis.cpp
    shell/test_lme_basis_curved.cpp
    shell/test_lme_boundary.cpp
    shell/test_lme_chart_connectivity.cpp
    shell/test_lme_curved_K.cpp
    shell/test_lme_curved_M.cpp
    shell/test_lme_curved_sphere_modes.cpp
    shell/test_lme_nodal_gaps.cpp
    shell/test_lme_patch.cpp
    shell/test_lme_shepard.cpp
    shell/test_lme_sme_basis.cpp
    shell/test_lme_sme_chart_diag.cpp
    shell/test_sme_brezel_roughness.cpp
    shell/test_sme_hemi_corner_dissection.cpp
    shell/test_sme_rim_feasibility_dissection.cpp
    shell/test_lme_free_edge_circular_plate.cpp
    shell/test_lme_ss_plate.cpp
    shell/test_loop_cylinder_doublet_basis.cpp
    shell/test_loop_cylinder_split_convergence.cpp
    shell/test_loop_cylinder_symmetry.cpp
    shell/test_loop_disk_method_table.cpp
    shell/test_loop_quadrature.cpp
    shell/test_loop_quadrature_sweep.cpp
    shell/test_loop_stam_eigenstructure.cpp
    shell/test_loop_stam_element_K.cpp
    shell/test_loop_stam_nan_repro.cpp
    shell/test_loop_stam_phi.cpp
    shell/test_loop_stam_phi_derivatives.cpp
    shell/test_loop_stam_picking_matrices.cpp
    shell/test_loop_stam_tile_map.cpp
    shell/test_loop_stencil.cpp
    shell/test_loop_subdivide.cpp
    shell/test_modes.cpp
    shell/test_modes_loop.cpp
    shell/test_modes_vs_annulus_clamped_analytic.cpp
    shell/test_modes_vs_complete_sphere_analytic.cpp
    shell/test_modes_vs_free_edge_circular_plate_analytic.cpp
    shell/test_modes_vs_free_free_cylinder_analytic.cpp
    shell/test_modes_vs_ss_rect_plate_analytic.cpp
    shell/test_rest_data.cpp
    shell/test_lme_evaluate_modes_projection.cpp
    shell/test_lme_membrane_locking_sri.cpp
    shell/test_rigid_filter_distribution.cpp
    shell/test_stam_quadrature_spike.cpp
    shell/test_static_obstacle_course.cpp
    shell/test_stiffness.cpp
    unit/test_mesh.cpp
    unit/test_mesh_alternate_diagonals.cpp
    unit/test_mesh_generate_cylinder.cpp
    unit/test_mesh_generate_disk_annulus.cpp
    unit/test_mesh_generate_flat_plate.cpp
    unit/test_mesh_generate_icosphere.cpp
    unit/test_synth.cpp
    unit/test_wav.cpp
)

# Make the source-tree models/ directory discoverable from tests at runtime.
target_compile_definitions(chladni_tests PRIVATE
    CHLADNI_DATA_DIR="${CMAKE_SOURCE_DIR}/models"
)

target_link_libraries(chladni_tests
    PRIVATE
        chladni::chladni
        chladni_warnings
        Catch2::Catch2WithMain
        # Spectra is header-only and already fetched at the top level.
        # Tests need it for BC-constrained eigensolves (clamped annulus
        # etc.) — the production `compute_shell_modes_loop` wraps a
        # rigid-body filter that doesn't apply to clamped problems
        # (K is positive definite there, no rigid kernel to filter).
        Spectra
)

# Register every Catch2 TEST_CASE as an individual CTest test
catch_discover_tests(chladni_tests)
