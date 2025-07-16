#!/bin/bash

# Build script for Discontinuous Galerkin Framework
# This script builds the project and runs all tests
# Usage: ./build.sh [--clean]

set -e  # Exit on any error

# Parse command line arguments
CLEAN_BUILD=false
CLEAN_TESTS=false
if [ "$1" == "--clean" ]; then
    CLEAN_BUILD=true
    CLEAN_TESTS=true
    shift
elif [ "$1" == "--clean-tests" ]; then
    CLEAN_TESTS=true
    shift
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --clean         Force a clean rebuild (removes all build directories)"
    echo "  --clean-tests   Clean only test build directory"
    echo "  --help          Show this help message"
    echo ""
    exit 0
fi

echo "================================================"
echo "Discontinuous Galerkin Framework Build Script"
echo "================================================"

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[STATUS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check if we're in the correct directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Check for required dependencies
print_info "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    print_error "CMake is required but not installed."
    exit 1
fi

if ! command -v make &> /dev/null; then
    print_error "Make is required but not installed."
    exit 1
fi

# Create build directory with smarter clean logic
print_status "Setting up build directory..."
if [ "$CLEAN_BUILD" == "true" ]; then
    print_info "Clean build requested, removing existing build directory..."
    rm -rf build
fi
mkdir -p build
cd build

# More intelligent reconfiguration check
NEED_CONFIGURE=false
if [ ! -f "Makefile" ] || [ "$CLEAN_BUILD" == "true" ]; then
    NEED_CONFIGURE=true
elif [ "../CMakeLists.txt" -nt "CMakeCache.txt" ]; then
    print_info "CMakeLists.txt is newer than cache, reconfiguring..."
    NEED_CONFIGURE=true
fi

if [ "$NEED_CONFIGURE" == "true" ]; then
    print_status "Configuring with CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
else
    print_info "Build already configured and up to date"
fi

# Build only if needed
print_status "Building project..."
make -j$(nproc)

cd ..

# Smarter test build logic
print_status "Setting up test build..."
cd tests
if [ "$CLEAN_TESTS" == "true" ]; then
    print_info "Cleaning test build directory..."
    rm -rf build
fi
mkdir -p build
cd build

# Only reconfigure tests if really needed
TEST_NEED_CONFIGURE=false
if [ ! -f "Makefile" ] || [ "$CLEAN_TESTS" == "true" ]; then
    TEST_NEED_CONFIGURE=true
elif [ "../CMakeLists.txt" -nt "CMakeCache.txt" ]; then
    TEST_NEED_CONFIGURE=true
fi

if [ "$TEST_NEED_CONFIGURE" == "true" ]; then
    print_status "Configuring test build..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
else
    print_info "Tests already configured and up to date"
fi

print_status "Building tests..."
make -j$(nproc)

# Go back to project root
cd ..

# Run all tests
print_status "Running comprehensive test suite..."
echo ""

# Run individual test files and collect results
test_files=("test_euler" "test_legendre" "test_gauss_legendre" "test_field" "test_rusanov" "test_dg_scheme" "test_runge_kutta" "test_lagrange" "test_mesh")

total_tests=0
passed_tests=0
failed_tests=0

for test in "${test_files[@]}"; do
    echo "Running $test..."
    if [ -f "./$test" ]; then
        # Check if test executable is newer than source files
        test_source="../test_${test#test_}.cpp"
        if [ -f "$test_source" ] && [ "./$test" -nt "$test_source" ]; then
            print_info "$test is up to date"
        else
            print_info "$test may need rebuilding"
        fi
        
        if ./$test; then
            echo "✅ $test PASSED"
            ((passed_tests++))
        else
            echo "❌ $test FAILED"
            ((failed_tests++))
        fi
    else
        print_warning "Test executable $test not found, skipping..."
        ((failed_tests++))
    fi
    ((total_tests++))
    echo ""
done

# Print summary
echo "================================================"
echo "BUILD AND TEST SUMMARY"
echo "================================================"
if [ "$CLEAN_BUILD" == "true" ]; then
    echo "Build type: Clean build (full rebuild)"
else
    echo "Build type: Incremental build (only changed files)"
fi
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"

if [ $failed_tests -eq 0 ]; then
    print_status "🎉 ALL TESTS PASSED! Build successful! 🎉"
    exit 0
else
    print_error "❌ $failed_tests test(s) failed. Please check the output above."
    exit 1
fi
