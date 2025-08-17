# Mixed Precision Unit Tests for LAMMPS KOKKOS (PR 4608)

## Overview

This directory contains comprehensive unit tests for PR 4608, which adds single precision (FP32) and mixed precision support to the LAMMPS KOKKOS package.

## Test Organization

The tests are organized into 14 groups to manage the large scope of changes (~170 files modified):

### Implemented Tests

1. **test_mixed_precision_types.cpp** - Core type system and infrastructure
   - Tests KK_FLOAT and KK_SUM_FLOAT type definitions
   - TransformView functionality
   - ArrayTypes precision-aware typedefs
   - Memory allocation with MemoryKokkos
   - View layouts and precision conversions

2. **test_mixed_precision_angles.cpp** - Angle styles
   - AngleHarmonicKokkos
   - AngleCharmmKokkos with Urey-Bradley
   - AngleCosineKokkos
   - AngleClass2Kokkos with cross terms
   - Force accumulation and energy/virial calculations

### Planned Tests (To Be Implemented)

3. **test_mixed_precision_bonds.cpp** - Bond styles
4. **test_mixed_precision_pairs_simple.cpp** - Simple pair styles
5. **test_mixed_precision_pairs_complex.cpp** - Complex many-body potentials
6. **test_mixed_precision_dihedrals.cpp** - Dihedral and improper styles
7. **test_mixed_precision_atomvec.cpp** - Atom and AtomVec classes
8. **test_mixed_precision_fixes_common.cpp** - Common fixes (NVE, NVT, NPT)
9. **test_mixed_precision_fixes_special.cpp** - Specialized fixes
10. **test_mixed_precision_computes.cpp** - Compute styles
11. **test_mixed_precision_neighbor.cpp** - Neighbor list and domain
12. **test_mixed_precision_comm.cpp** - Communication and FFT
13. **test_mixed_precision_kspace.cpp** - KSPACE integration
14. **test_mixed_precision_build.cpp** - Build system and configuration

## Building and Running Tests

### Prerequisites

- LAMMPS built with KOKKOS package enabled
- GoogleTest framework
- MPI support

### Building with Different Precision Modes

The tests support three precision modes controlled by CMake variables:

#### 1. Double Precision (Default)
```bash
cmake -D PKG_KOKKOS=ON \
      -D KOKKOS_PREC=double \
      -D BUILD_TESTING=ON \
      ../cmake
make
```

#### 2. Mixed Precision (FP32 compute, FP64 accumulation)
```bash
cmake -D PKG_KOKKOS=ON \
      -D KOKKOS_PREC=mixed \
      -D BUILD_TESTING=ON \
      ../cmake
make
```

#### 3. Single Precision (FP32 only)
```bash
cmake -D PKG_KOKKOS=ON \
      -D KOKKOS_PREC=single \
      -D BUILD_TESTING=ON \
      ../cmake
make
```

### Running Tests

Run all mixed precision tests:
```bash
ctest -R MixedPrecision
```

Run specific test group:
```bash
ctest -R MixedPrecisionTypes     # Core types only
ctest -R MixedPrecisionAngles    # Angle styles only
```

Run with verbose output:
```bash
ctest -V -R MixedPrecision
```

### Testing with Different Backends

#### CPU (OpenMP)
```bash
cmake -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D KOKKOS_PREC=mixed \
      ../cmake
```

#### CUDA GPU
```bash
cmake -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D CMAKE_CXX_COMPILER=$KOKKOS_PATH/bin/nvcc_wrapper \
      -D KOKKOS_PREC=mixed \
      ../cmake
```

#### HIP GPU (AMD)
```bash
cmake -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_HIP=ON \
      -D KOKKOS_PREC=mixed \
      ../cmake
```

## Test Coverage

Each test file covers:

1. **Type Correctness** - Verifies correct precision types are used
2. **Numerical Accuracy** - Checks calculations meet precision tolerances
3. **Memory Management** - Tests view creation, syncing, and destruction
4. **Edge Cases** - Tests extreme values, near-zero angles, etc.
5. **Performance** - Ensures no unexpected precision-related slowdowns

## Precision Tolerances

The tests use adaptive tolerances based on precision mode:

- **Double Precision**: Relative tolerance 1e-12, Absolute tolerance 1e-14
- **Mixed Precision**: Relative tolerance 1e-6, Absolute tolerance 1e-7
- **Single Precision**: Relative tolerance 1e-5, Absolute tolerance 1e-6

## Utilities

**test_mixed_precision_utils.h** provides common testing utilities:

- `MixedPrecisionTestFixture` - Base test fixture class
- `approxEqual()` - Precision-aware floating point comparison
- `checkNumericalStability()` - Detects NaN/Inf values
- `createLAMMPSInstance()` - Creates LAMMPS with proper Kokkos settings
- Precision mode detection macros

## Continuous Integration

For CI/CD pipelines, run all three precision modes:

```bash
#!/bin/bash
# Test all precision modes
for prec in double mixed single; do
    echo "Testing $prec precision..."
    cmake -D KOKKOS_PREC=$prec -D PKG_KOKKOS=ON -D BUILD_TESTING=ON ../cmake
    make -j
    ctest -R MixedPrecision --output-on-failure
done
```

## Known Issues and Limitations

1. Some tests may need to be skipped in single precision mode due to numerical stability
2. GPU tests require appropriate hardware and drivers
3. Mixed precision mode provides a balance between speed and accuracy

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use the utilities in `test_mixed_precision_utils.h`
3. Test all three precision modes
4. Document any precision-specific behaviors
5. Keep test files under 1000 lines for maintainability

## References

- [PR 4608](https://github.com/lammps/lammps/pull/4608) - Original pull request
- [KOKKOS Documentation](https://kokkos.github.io/kokkos-core-wiki/) - Kokkos programming guide
- [LAMMPS KOKKOS Package](https://docs.lammps.org/Speed_kokkos.html) - Package documentation
