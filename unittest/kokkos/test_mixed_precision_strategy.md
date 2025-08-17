# Unit Test Strategy for PR 4608: KOKKOS Mixed Precision Support

## Overview
PR 4608 adds FP32 and mixed precision support to LAMMPS KOKKOS package by:
- Introducing new precision types (KK_FLOAT, KK_SUM_FLOAT, etc.)
- Adding TransformView templates for automatic type conversion
- Modifying all KOKKOS styles to use precision-aware types
- Adding CMake/build system support for precision selection

## Test File Organization Strategy

### Group 1: Core Type System & Infrastructure
**File: test_mixed_precision_types.cpp**
- Test KK_FLOAT, KK_SUM_FLOAT type definitions for different precision modes
- Test TransformView functionality and automatic conversions
- Test ArrayTypes precision-aware typedefs
- Test kokkos_type.h macros and precision settings
- Coverage: kokkos_type.h, kokkos_base.h, atom_kokkos.h type changes

### Group 2: Angle Styles
**File: test_mixed_precision_angles.cpp**
- AngleCharmmKokkos precision changes
- AngleClass2Kokkos precision changes  
- AngleCosineKokkos precision changes
- AngleHarmonicKokkos precision changes
- AngleSPICAKokkos precision changes
- AngleHybridKokkos changes
- Coverage: angle_*_kokkos.cpp/h files

### Group 3: Bond Styles
**File: test_mixed_precision_bonds.cpp**
- BondClass2Kokkos precision changes
- BondFENEKokkos precision changes
- BondHarmonicKokkos precision changes
- BondHybridKokkos changes
- BondMorseKokkos precision changes
- Coverage: bond_*_kokkos.cpp/h files

### Group 4: Pair Styles (Part 1 - Simple)
**File: test_mixed_precision_pairs_simple.cpp**
- PairLJCutKokkos precision changes
- PairLJCutCoulCutKokkos precision changes
- PairMorseKokkos precision changes
- PairBuckKokkos precision changes
- PairYukawaKokkos precision changes
- Coverage: simpler pair styles

### Group 5: Pair Styles (Part 2 - Complex)
**File: test_mixed_precision_pairs_complex.cpp**
- PairEAMKokkos precision changes
- PairSWKokkos precision changes
- PairTersoffKokkos precision changes
- PairREBOKokkos precision changes
- Coverage: complex many-body potentials

### Group 6: Dihedral & Improper Styles
**File: test_mixed_precision_dihedrals.cpp**
- DihedralCharmmKokkos precision changes
- DihedralClass2Kokkos precision changes
- DihedralHarmonicKokkos precision changes
- DihedralOPLSKokkos precision changes
- ImproperClass2Kokkos precision changes
- ImproperHarmonicKokkos precision changes
- Coverage: dihedral_*_kokkos.cpp/h, improper_*_kokkos.cpp/h

### Group 7: Atom & AtomVec Classes
**File: test_mixed_precision_atomvec.cpp**
- AtomKokkos array type changes
- AtomVecAngleKokkos precision changes
- AtomVecAtomicKokkos precision changes
- AtomVecBondKokkos precision changes
- AtomVecChargeKokkos precision changes
- AtomVecFullKokkos precision changes
- AtomVecMolecularKokkos precision changes
- Coverage: atom_kokkos.cpp/h, atom_vec_*_kokkos.cpp/h

### Group 8: Fix Styles (Part 1 - Common)
**File: test_mixed_precision_fixes_common.cpp**
- FixNVEKokkos precision changes
- FixNVTKokkos precision changes
- FixNPTKokkos precision changes
- FixLangevinKokkos precision changes
- FixSetForceKokkos precision changes
- Coverage: common time integration fixes

### Group 9: Fix Styles (Part 2 - Specialized)
**File: test_mixed_precision_fixes_special.cpp**
- FixDPDKokkos precision changes
- FixShakeKokkos precision changes
- FixRigidKokkos precision changes
- FixWallKokkos precision changes
- Coverage: specialized constraint/wall fixes

### Group 10: Compute Styles
**File: test_mixed_precision_computes.cpp**
- ComputeTempKokkos precision changes
- ComputePressureKokkos precision changes
- ComputePEKokkos precision changes
- ComputeRDFKokkos precision changes
- Coverage: compute_*_kokkos.cpp/h files

### Group 11: Neighbor List & Domain
**File: test_mixed_precision_neighbor.cpp**
- NeighborKokkos precision changes
- NPairKokkos precision changes
- NStencilKokkos precision changes
- DomainKokkos precision changes
- Coverage: neighbor_kokkos.cpp/h, npair_kokkos.cpp/h, domain_kokkos.cpp/h

### Group 12: Communication & FFT
**File: test_mixed_precision_comm.cpp**
- CommKokkos precision changes
- CommTiledKokkos precision changes
- FFTKokkos precision changes
- GridCommKokkos precision changes
- Coverage: comm_kokkos.cpp/h, comm_tiled_kokkos.cpp/h, fft*_kokkos.cpp/h

### Group 13: KSPACE Package Integration
**File: test_mixed_precision_kspace.cpp**
- PPPMKokkos precision changes
- EwaldKokkos precision changes
- MSMKokkos precision changes
- Coverage: kspace-related kokkos files

### Group 14: Build System & Configuration
**File: test_mixed_precision_build.cpp**
- Test CMake KOKKOS_PREC settings
- Test KOKKOS_LAYOUT settings
- Test preprocessor macros (LMP_KOKKOS_DOUBLE_DOUBLE, etc.)
- Coverage: cmake/Modules/Packages/KOKKOS.cmake

## Implementation Priority

1. **Start with Group 1** (Core Types) - Foundation for all other tests
2. **Then Groups 2-6** (Force field styles) - Most lines of code changed
3. **Then Group 7** (AtomVec) - Critical data structure changes
4. **Then Groups 8-10** (Fixes/Computes) - User-facing functionality
5. **Finally Groups 11-14** (Infrastructure) - Supporting systems

## Testing Approach for Each Group

Each test file should:
1. Test all three precision modes (double, mixed, single)
2. Verify type sizes and alignments
3. Test data conversion between precisions
4. Test atomic operations with new types
5. Verify numerical accuracy for each precision mode
6. Test view syncing between Host/Device with new types
7. Test MPI communication with precision-aware types

## Key Testing Macros/Functions Needed

```cpp
// Test different precision configurations
#define TEST_ALL_PRECISIONS(test_func) \
  TEST(MixedPrecision, test_func##_Double) { \
    test_with_precision<DOUBLE_DOUBLE>(#test_func); \
  } \
  TEST(MixedPrecision, test_func##_Mixed) { \
    test_with_precision<SINGLE_DOUBLE>(#test_func); \
  } \
  TEST(MixedPrecision, test_func##_Single) { \
    test_with_precision<SINGLE_SINGLE>(#test_func); \
  }
```

## Estimated Lines per Test File
- Each test file: ~500-1000 lines
- Total: ~7,000-14,000 lines of test code
- This ensures comprehensive coverage while keeping each file manageable

## Next Steps
1. Create test_mixed_precision_types.cpp first (foundation)
2. Implement helper utilities for precision testing
3. Systematically work through each group
4. Run coverage analysis after each group to ensure completeness
