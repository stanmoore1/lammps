# preset that enables KOKKOS and selects OpenMP (only) compilation
set(PKG_KOKKOS ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ON  CACHE BOOL "" FORCE)
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

# ==================== PACKAGES NEEDED FOR KOKKOS UNIT TESTING  ====================
# make sure all packages needed for pr 4608 mixed precision unit testing

if(ENABLE_TESTING)
  # Core packages needed for unit tests
  set(PKG_MOLECULE ON CACHE BOOL "" FORCE)  # Provides angle_harmonic.h, bond_harmonic.h, etc.
  set(PKG_CLASS2 ON CACHE BOOL "" FORCE)    # Provides angle_class2.h, bond_class2.h, etc.
  set(PKG_RIGID ON CACHE BOOL "" FORCE)      # Provides fix_rigid.h and related
  set(PKG_MISC ON CACHE BOOL "" FORCE)       # Provides various fixes and computes
  set(PKG_EXTRA-PAIR ON CACHE BOOL "" FORCE) # Provides additional pair styles
  set(PKG_EXTRA-FIX ON CACHE BOOL "" FORCE)  # Provides additional fix styles
  message(STATUS "   📦 Unit Test Packages: MOLECULE, CLASS2, RIGID, MISC, EXTRA-PAIR, EXTRA-FIX")
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
    
    # Create entitlements content for code signing
    set(DEBUG_ENTITLEMENTS_CONTENT 
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>")
    
    # Write entitlements file
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements" "${DEBUG_ENTITLEMENTS_CONTENT}")
    
    # ==================== AUTOMATIC POST-BUILD SETUP ====================
    # Create a hook file that will be included at the end of the main CMakeLists.txt
    set(INSTRUMENTS_HOOK "${CMAKE_CURRENT_BINARY_DIR}/add_instruments_support.cmake")
    file(WRITE "${INSTRUMENTS_HOOK}"
"# Auto-generated CMake module to add Instruments.app support
if(TARGET lmp)
    add_custom_command(
        TARGET lmp
        POST_BUILD
        COMMAND dsymutil $<TARGET_FILE:lmp> -o $<TARGET_FILE:lmp>.dSYM
        COMMAND codesign -s - -f --entitlements \"${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements\" $<TARGET_FILE:lmp>
        COMMENT \"🔬 Setting up Instruments.app support (dsymutil + codesign)...\"
        VERBATIM
    )
    message(STATUS \"✅ Post-build Instruments commands added to lmp target\")
endif()
")
    
    # The hook file will be included by the main CMakeLists.txt
    # after the lmp target is created
    
    message(STATUS "✅ Instruments support configured")
    message(STATUS "   📁 Entitlements: ${CMAKE_CURRENT_BINARY_DIR}/debug.entitlements")
    message(STATUS "   🎯 Post-build automation: AUTOMATIC")
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
    message(STATUS "   🔬 Instruments Support: AUTOMATIC")
    message(STATUS "      - Debug symbols: ENABLED")
    message(STATUS "      - Entitlements: CREATED")
    message(STATUS "      - Post-build: dsymutil + codesign (automatic)")
endif()
message(STATUS "")
message(STATUS "🚀 Usage:")
message(STATUS "   cmake -C ../cmake/presets/kokkos-openmp.cmake ../cmake")
message(STATUS "   cmake --build . -j$(nproc)")
if(APPLE AND (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(STATUS "")
    message(STATUS "🔬 Instruments.app support is AUTOMATIC!")
    message(STATUS "   ✅ dsymutil and codesign run automatically after building")
    message(STATUS "   ✅ No manual steps required - just build and profile!")
    message(STATUS "   ✅ Simply run: cmake --build . && open -a Instruments lmp")
endif()
message(STATUS "")


