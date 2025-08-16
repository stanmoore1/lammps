# preset that enables KOKKOS and selects OpenMP (only) compilation
set(PKG_KOKKOS ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ON  CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
set(BUILD_OMP ON CACHE BOOL "" FORCE)

# hide deprecation warnings temporarily for stable release
set(Kokkos_ENABLE_DEPRECATION_WARNINGS OFF CACHE BOOL "" FORCE)

# ==================== KOKKOS DEBUG SETTINGS ====================
# Enable comprehensive Kokkos debugging for Debug builds
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    
    # Kokkos debugging and bounds checking
    set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Enable Kokkos debugging" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Enable Kokkos bounds checking" FORCE)
    set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL "Enable DualView modify checking" FORCE)
    
    # Additional Kokkos debug features
    set(Kokkos_ENABLE_PROFILING ON CACHE BOOL "Enable Kokkos profiling" FORCE)
    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Disable aggressive vectorization for debugging" FORCE)
    
    message(STATUS "🔍 Kokkos Debug mode enabled with bounds checking and profiling")
endif()

# ==================== Robust OpenMP Configuration for macOS ====================
if(APPLE)
    message(STATUS "🍎 Configuring OpenMP for macOS...")
    
    # Use Apple Clang with Homebrew libomp
    find_program(CMAKE_C_COMPILER NAMES clang PATHS /opt/homebrew/bin /usr/bin)
    find_program(CMAKE_CXX_COMPILER NAMES clang++ PATHS /opt/homebrew/bin /usr/bin)
    if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
        message(FATAL_ERROR "Apple Clang not found.")
    endif()
    
    # Detect Homebrew libomp install prefix
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(NOT EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(FATAL_ERROR 
            "❌ libomp not found. Please install with:\n"
            "    brew install libomp\n"
            "\n"
            "If you already have libomp installed but this error persists,\n"
            "make sure Homebrew is in your PATH and try:\n"
            "    brew reinstall libomp")
    endif()
    
    message(STATUS "✅ Using Apple Clang with libomp from ${LIBOMP_PREFIX}")
    
    # Configure OpenMP flags for Apple Clang + Homebrew libomp
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE STRING "" FORCE)
    
    # Set include and library directories
    include_directories("${LIBOMP_PREFIX}/include")
    link_directories("${LIBOMP_PREFIX}/lib")
    
else()
    message(STATUS "🐧 Non-macOS system detected - using standard OpenMP detection")
endif()

# ==================== DEBUG SYMBOLS & INSTRUMENTS COMPATIBILITY ====================
if(APPLE AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(STATUS "🔬 Configuring Instruments.app compatibility...")
    
    # Comprehensive debug flags for Instruments compatibility
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline" CACHE STRING "" FORCE)
    
    # Generate debug symbols in separate file for better Instruments support
    set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym" CACHE STRING "" FORCE)
    set(CMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS YES CACHE STRING "" FORCE)
    
    # Create entitlements content and write file
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements" 
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>")
    
    # Create a CMake script to run post-build dSYM and codesigning
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/setup_instruments.cmake"
"# Instruments setup script - auto-generated
message(STATUS \"🔬 Setting up Instruments support...\")

# Setup for lmp executable
if(EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/lmp\")
    message(STATUS \"   📱 Processing lmp executable...\")
    execute_process(
        COMMAND dsymutil \"${CMAKE_CURRENT_BINARY_DIR}/lmp\" -o \"${CMAKE_CURRENT_BINARY_DIR}/lmp.dSYM\"
        RESULT_VARIABLE DSYM_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(DSYM_RESULT EQUAL 0)
        message(STATUS \"   ✅ dSYM generated: lmp.dSYM\")
    endif()
    
    execute_process(
        COMMAND codesign -s - -f --entitlements \"${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements\" \"${CMAKE_CURRENT_BINARY_DIR}/lmp\"
        RESULT_VARIABLE SIGN_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(SIGN_RESULT EQUAL 0)
        message(STATUS \"   🔐 Code signed: lmp\")
    endif()
    
    if(DSYM_RESULT EQUAL 0 AND SIGN_RESULT EQUAL 0)
        message(STATUS \"   ✅ lmp is ready for Instruments.app!\")
    endif()
else()
    message(STATUS \"   ⏳ lmp not found - skipping\")
endif()

# Setup for lammps library (if built as shared)
if(EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/liblammps.dylib\")
    message(STATUS \"   📚 Processing lammps library...\")
    execute_process(
        COMMAND dsymutil \"${CMAKE_CURRENT_BINARY_DIR}/liblammps.dylib\" -o \"${CMAKE_CURRENT_BINARY_DIR}/liblammps.dylib.dSYM\"
        RESULT_VARIABLE LIB_DSYM_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(LIB_DSYM_RESULT EQUAL 0)
        message(STATUS \"   ✅ dSYM generated: liblammps.dylib.dSYM\")
    endif()
    
    execute_process(
        COMMAND codesign -s - -f --entitlements \"${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements\" \"${CMAKE_CURRENT_BINARY_DIR}/liblammps.dylib\"
        RESULT_VARIABLE LIB_SIGN_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(LIB_SIGN_RESULT EQUAL 0)
        message(STATUS \"   🔐 Code signed: liblammps.dylib\")
    endif()
    
    if(LIB_DSYM_RESULT EQUAL 0 AND LIB_SIGN_RESULT EQUAL 0)
        message(STATUS \"   ✅ lammps library is ready for Instruments.app!\")
    endif()
else()
    message(STATUS \"   ⏳ lammps library not found - skipping\")
endif()
")
    
    # Add custom target that runs the setup script
    add_custom_target(instruments_setup
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/setup_instruments.cmake"
        COMMENT "🔬 Setting up Instruments support for LAMMPS"
        VERBATIM
    )
    
    message(STATUS "✅ Instruments support configured")
    message(STATUS "   📁 Entitlements: ${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements")
    message(STATUS "   📜 Setup script: ${CMAKE_CURRENT_BINARY_DIR}/setup_instruments.cmake")
endif()

# ==================== SUMMARY ====================
message(STATUS "")
message(STATUS "🎯 LAMMPS Kokkos + OpenMP Configuration Summary:")
message(STATUS "   📦 Kokkos: ENABLED (OpenMP backend)")
message(STATUS "   🧵 OpenMP: ENABLED")
message(STATUS "   🔍 Build Type: ${CMAKE_BUILD_TYPE}")
if(Kokkos_ENABLE_DEBUG)
    message(STATUS "   🐛 Kokkos Debug: ENABLED")
    message(STATUS "      - Bounds checking: ON")
    message(STATUS "      - DualView checking: ON") 
    message(STATUS "      - Profiling: ON")
endif()
if(APPLE AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(STATUS "   🔬 Instruments Support: CONFIGURED")
    message(STATUS "      - dSYM generation: READY")
    message(STATUS "      - Code signing: READY")
    message(STATUS "      - Debug entitlements: CREATED")
endif()
message(STATUS "")
message(STATUS "🚀 Usage:")
message(STATUS "   cmake -C ../cmake/presets/kokkos-openmp.cmake ../cmake")
message(STATUS "   cmake --build . -j$(nproc)")
if(APPLE AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(STATUS "   cmake --build . --target instruments_setup  # Setup for Instruments")
    message(STATUS "   # Then ready for Instruments.app profiling!")
endif()
message(STATUS "")
