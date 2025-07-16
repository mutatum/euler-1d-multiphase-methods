@echo off
REM Build script for Discontinuous Galerkin Framework (Windows)
REM This script builds the project and runs all tests

setlocal enabledelayedexpansion

echo ================================================
echo Discontinuous Galerkin Framework Build Script
echo ================================================

REM Check if we're in the correct directory
if not exist "CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found. Please run this script from the project root directory.
    exit /b 1
)

REM Check for required dependencies
echo [INFO] Checking dependencies...

where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] CMake is required but not installed.
    exit /b 1
)

REM Create build directory
echo [STATUS] Creating build directory...
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo [STATUS] Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

REM Build the project
echo [STATUS] Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

REM Go back to project root
cd ..

REM Build and run tests
echo [STATUS] Building and running tests...
cd tests
if not exist "build" mkdir build
cd build

REM Configure test build
cmake .. -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] Test configuration failed.
    exit /b 1
)

REM Build tests
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Test build failed.
    exit /b 1
)

REM Run all tests
echo [STATUS] Running comprehensive test suite...
echo.

REM Run individual test files and collect results
set test_files=test_euler test_legendre test_gauss_legendre test_field test_rusanov test_dg_scheme test_runge_kutta test_lagrange test_mesh

set total_tests=0
set passed_tests=0
set failed_tests=0

for %%t in (%test_files%) do (
    echo Running %%t...
    %%t.exe
    if errorlevel 1 (
        echo ❌ %%t FAILED
        set /a failed_tests+=1
    ) else (
        echo ✅ %%t PASSED
        set /a passed_tests+=1
    )
    set /a total_tests+=1
    echo.
)

REM Print summary
echo ================================================
echo BUILD AND TEST SUMMARY
echo ================================================
echo Total tests: %total_tests%
echo Passed: %passed_tests%
echo Failed: %failed_tests%

if %failed_tests% equ 0 (
    echo [STATUS] 🎉 ALL TESTS PASSED! Build successful! 🎉
    exit /b 0
) else (
    echo [ERROR] ❌ %failed_tests% test(s) failed. Please check the output above.
    exit /b 1
)
