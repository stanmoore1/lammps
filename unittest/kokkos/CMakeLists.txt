# -*- CMake -*- file for KOKKOS testing

# Mixed precision tests for PR 4608 (KOKKOS FP32 support)
if(PKG_KOKKOS)
  # Core type system tests
  add_executable(test_mixed_precision_types test_mixed_precision_types.cpp)
  target_link_libraries(test_mixed_precision_types PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_types PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionTypes COMMAND test_mixed_precision_types)
  set_tests_properties(MixedPrecisionTypes PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Simple pair styles tests
  add_executable(test_mixed_precision_pairs_simple test_mixed_precision_pairs_simple.cpp)
  target_link_libraries(test_mixed_precision_pairs_simple PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_pairs_simple PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionPairsSimple COMMAND test_mixed_precision_pairs_simple)
  set_tests_properties(MixedPrecisionPairsSimple PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Complex many-body pair styles tests (EAM, SW, Tersoff, REBO)
  add_executable(test_mixed_precision_pairs_complex test_mixed_precision_pairs_complex.cpp)
  target_include_directories(test_mixed_precision_pairs_complex PRIVATE ${LAMMPS_SOURCE_DIR}/MANYBODY)
  target_link_libraries(test_mixed_precision_pairs_complex PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_pairs_complex PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionPairsComplex COMMAND test_mixed_precision_pairs_complex)
  set_tests_properties(MixedPrecisionPairsComplex PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # AtomVec tests
  add_executable(test_mixed_precision_atomvec test_mixed_precision_atomvec.cpp)
  target_include_directories(test_mixed_precision_atomvec PRIVATE ${LAMMPS_SOURCE_DIR}/MOLECULE)
  target_link_libraries(test_mixed_precision_atomvec PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_atomvec PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionAtomVec COMMAND test_mixed_precision_atomvec)
  set_tests_properties(MixedPrecisionAtomVec PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Common fixes tests (NVE, NVT, NPT, etc.)
  add_executable(test_mixed_precision_fixes_common test_mixed_precision_fixes_common.cpp)
  target_link_libraries(test_mixed_precision_fixes_common PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_fixes_common PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionFixesCommon COMMAND test_mixed_precision_fixes_common)
  set_tests_properties(MixedPrecisionFixesCommon PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Specialized fixes tests (SHAKE, Rigid, Wall, DPD)
  # Note: Some tests require specific packages (RIGID, DPD, etc.)
  if(PKG_RIGID OR PKG_DPD)
    add_executable(test_mixed_precision_fixes_special test_mixed_precision_fixes_special.cpp)
    target_include_directories(test_mixed_precision_fixes_special PRIVATE
      ${LAMMPS_SOURCE_DIR}/DPD
      ${LAMMPS_SOURCE_DIR}/RIGID
    )
    target_link_libraries(test_mixed_precision_fixes_special PRIVATE lammps kokkos GTest::GTestMain)
    target_compile_definitions(test_mixed_precision_fixes_special PRIVATE -DLMP_KOKKOS)
    add_test(NAME MixedPrecisionFixesSpecial COMMAND test_mixed_precision_fixes_special)
    set_tests_properties(MixedPrecisionFixesSpecial PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  endif()

  if(PKG_MOLECULE)

    # Angle styles tests
    add_executable(test_mixed_precision_angles test_mixed_precision_angles.cpp)
    target_include_directories(test_mixed_precision_angles PRIVATE
      ${LAMMPS_SOURCE_DIR}/CLASS2
      ${LAMMPS_SOURCE_DIR}/MOLECULE
    )
    target_link_libraries(test_mixed_precision_angles PRIVATE lammps kokkos GTest::GTestMain)
    target_compile_definitions(test_mixed_precision_angles PRIVATE -DLMP_KOKKOS)
    add_test(NAME MixedPrecisionAngles COMMAND test_mixed_precision_angles)
    set_tests_properties(MixedPrecisionAngles PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
    # Bond styles tests
    add_executable(test_mixed_precision_bonds test_mixed_precision_bonds.cpp)
    target_include_directories(test_mixed_precision_bonds PRIVATE ${LAMMPS_SOURCE_DIR}/MOLECULE)
    target_link_libraries(test_mixed_precision_bonds PRIVATE lammps kokkos GTest::GTestMain)
    target_compile_definitions(test_mixed_precision_bonds PRIVATE -DLMP_KOKKOS)
    add_test(NAME MixedPrecisionBonds COMMAND test_mixed_precision_bonds)
    set_tests_properties(MixedPrecisionBonds PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
    # Dihedral and improper styles tests
    add_executable(test_mixed_precision_dihedrals test_mixed_precision_dihedrals.cpp)
    target_include_directories(test_mixed_precision_dihedrals PRIVATE
      ${LAMMPS_SOURCE_DIR}/CLASS2
      ${LAMMPS_SOURCE_DIR}/MOLECULE
    )
    target_link_libraries(test_mixed_precision_dihedrals PRIVATE lammps kokkos GTest::GTestMain)
    target_compile_definitions(test_mixed_precision_dihedrals PRIVATE -DLMP_KOKKOS)
    add_test(NAME MixedPrecisionDihedrals COMMAND test_mixed_precision_dihedrals)
    set_tests_properties(MixedPrecisionDihedrals PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  endif()
  
  # Compute styles tests
  add_executable(test_mixed_precision_computes test_mixed_precision_computes.cpp)
  target_link_libraries(test_mixed_precision_computes PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_computes PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionComputes COMMAND test_mixed_precision_computes)
  set_tests_properties(MixedPrecisionComputes PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")

endif()
