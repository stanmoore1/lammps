# -*- CMake -*- file for KOKKOS testing

# Mixed precision tests for PR 4608 (KOKKOS FP32 support)
if(PKG_KOKKOS)
  # Core type system tests
  add_executable(test_mixed_precision_types test_mixed_precision_types.cpp)
  target_link_libraries(test_mixed_precision_types PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_types PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionTypes COMMAND test_mixed_precision_types)
  set_tests_properties(MixedPrecisionTypes PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Angle styles tests
  add_executable(test_mixed_precision_angles test_mixed_precision_angles.cpp)
  target_link_libraries(test_mixed_precision_angles PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_angles PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionAngles COMMAND test_mixed_precision_angles)
  set_tests_properties(MixedPrecisionAngles PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Bond styles tests
  add_executable(test_mixed_precision_bonds test_mixed_precision_bonds.cpp)
  target_link_libraries(test_mixed_precision_bonds PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_bonds PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionBonds COMMAND test_mixed_precision_bonds)
  set_tests_properties(MixedPrecisionBonds PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Simple pair styles tests
  add_executable(test_mixed_precision_pairs_simple test_mixed_precision_pairs_simple.cpp)
  target_link_libraries(test_mixed_precision_pairs_simple PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_pairs_simple PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionPairsSimple COMMAND test_mixed_precision_pairs_simple)
  set_tests_properties(MixedPrecisionPairsSimple PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # AtomVec tests
  add_executable(test_mixed_precision_atomvec test_mixed_precision_atomvec.cpp)
  target_link_libraries(test_mixed_precision_atomvec PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_atomvec PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionAtomVec COMMAND test_mixed_precision_atomvec)
  set_tests_properties(MixedPrecisionAtomVec PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Dihedral and improper styles tests
  add_executable(test_mixed_precision_dihedrals test_mixed_precision_dihedrals.cpp)
  target_link_libraries(test_mixed_precision_dihedrals PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_dihedrals PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionDihedrals COMMAND test_mixed_precision_dihedrals)
  set_tests_properties(MixedPrecisionDihedrals PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
  
  # Common fixes tests (NVE, NVT, NPT, etc.)
  add_executable(test_mixed_precision_fixes_common test_mixed_precision_fixes_common.cpp)
  target_link_libraries(test_mixed_precision_fixes_common PRIVATE lammps kokkos GTest::GTestMain)
  target_compile_definitions(test_mixed_precision_fixes_common PRIVATE -DLMP_KOKKOS)
  add_test(NAME MixedPrecisionFixesCommon COMMAND test_mixed_precision_fixes_common)
  set_tests_properties(MixedPrecisionFixesCommon PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=7;OMP_PROC_BIND=false")
endif()
