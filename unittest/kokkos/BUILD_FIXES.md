# Build Fixes for Mixed Precision Unit Tests

## Fixed Issues

### 1. test_mixed_precision_neighbor.cpp
**Issue**: References to non-existent member variables in NeighborKokkos and DomainKokkos classes
- `k_bboxlo`, `k_bboxhi` - These Kokkos views don't exist in NeighborKokkos
- `k_boxlo`, `k_boxhi`, `k_prd` - These Kokkos views don't exist in DomainKokkos  
- `k_numneigh`, `k_neighbors` - Should use `d_numneigh` and `d_neighbors` instead
- `nstencil_list` - This array doesn't exist; stencil data is internal to NPair classes

**Fix**: Removed references to non-existent members and updated to use correct member names where applicable.

### 2. test_mixed_precision_pairs_complex.cpp  
**Issue**: Linker errors for missing PairSW and PairEAM base class symbols
- These classes are part of the MANYBODY package which may not be compiled

**Fix**: Added conditional compilation using `#ifdef LMP_MANYBODY` to skip tests when the package is not available. Also updated CMakeLists.txt to properly detect and define the macro.

### 3. test_mixed_precision_fixes_special.cpp
**Issue**: FixRigidKokkos template class doesn't exist
- Tests were trying to cast to `FixRigidKokkos<LMPDeviceType>*` which doesn't exist

**Fix**: Commented out tests that rely on FixRigidKokkos (tests 10 and 13) as this class doesn't exist as a template in the current LAMMPS codebase.

### 4. CMakeLists.txt
**Issue**: Missing conditional compilation for MANYBODY package

**Fix**: Added proper conditional check for PKG_MANYBODY and corresponding compile definitions.

## Required Packages for Full Test Coverage

To build and run all tests successfully, the following LAMMPS packages need to be enabled:

### Core Requirements
- **PKG_KOKKOS** - Essential (all tests require this)
- **BUILD_TESTING** - Required to build unit tests

### Package Dependencies
- **PKG_MOLECULE** - Required for bond, angle, dihedral tests
- **PKG_CLASS2** - Required for Class2 force field tests  
- **PKG_MANYBODY** - Required for EAM, SW, Tersoff pair style tests
- **PKG_RIGID** - Required for rigid body tests (though some tests are disabled)
- **PKG_DPD-BASIC** or **PKG_DPD-REACT** - Required for DPD tests
- **PKG_KSPACE** - Required for PPPM, Ewald, MSM tests
- **PKG_MISC** - Provides various fixes and computes
- **PKG_EXTRA-PAIR** - Provides additional pair styles
- **PKG_EXTRA-FIX** - Provides additional fix styles

## Build Instructions

### Minimal Build (Core tests only)
```bash
cmake -C ../cmake/presets/kokkos-openmp.cmake \
      -D PKG_KOKKOS=ON \
      -D BUILD_TESTING=ON \
      ../cmake
make -j
```

### Full Build (All tests)
```bash
cmake -C ../cmake/presets/kokkos-openmp.cmake \
      -D PKG_KOKKOS=ON \
      -D PKG_MOLECULE=ON \
      -D PKG_CLASS2=ON \
      -D PKG_MANYBODY=ON \
      -D PKG_RIGID=ON \
      -D PKG_DPD-BASIC=ON \
      -D PKG_KSPACE=ON \
      -D PKG_MISC=ON \
      -D PKG_EXTRA-PAIR=ON \
      -D PKG_EXTRA-FIX=ON \
      -D BUILD_TESTING=ON \
      -D ENABLE_TESTING=ON \
      ../cmake
make -j
```

## Running Tests

After building, run the tests with:
```bash
# Run all mixed precision tests
ctest -R MixedPrecision

# Run with verbose output to see failures
ctest -V -R MixedPrecision

# Run specific test groups
ctest -R MixedPrecisionNeighbor
ctest -R MixedPrecisionPairsComplex
ctest -R MixedPrecisionFixesSpecial
```

## Known Limitations

1. **FixRigidKokkos Tests**: Some rigid body tests are disabled because FixRigidKokkos doesn't exist as a template class. The rigid body functionality may be implemented differently in Kokkos.

2. **MANYBODY Package**: When MANYBODY package is not available, the complex pair tests will be skipped with a GTEST_SKIP message.

3. **Precision-Dependent Tolerances**: Some tests may need tolerance adjustments based on the precision mode (single, mixed, or double).

## Next Steps

1. Verify that the fixes allow successful compilation
2. Run the tests to ensure they pass
3. Consider implementing additional tests for missing coverage areas
4. Update documentation based on actual test results
