cmake_minimum_required(VERSION 3.10)

project(DGProject CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Eigen ---
# Add the Eigen subdirectory. This will define the Eigen3::Eigen target.
# The second argument tells CMake where to find the build output, 
# so it doesn't clutter the Eigen source directory.
add_subdirectory(external/eigen-3.4.0 ${CMAKE_BINARY_DIR}/eigen-build)

# --- Project Sources ---
# Add the 'src' directory to the include path so you can use #include "..."
# for headers in src.
# include_directories(src) <-- This is old-style, we'll use the modern target-based approach.

# Create a library from your source code in the 'src' directory.
# Since the 'src' directory only contains headers, we create an INTERFACE library.
# This is a "virtual" library that doesn't compile to anything, but bundles
# usage requirements (like include directories and dependencies) for other targets.
add_library(dg_lib INTERFACE)

# Add the 'src' directory to the public include directories of our interface library.
# Any target that links to dg_lib will automatically get this include directory.
target_include_directories(dg_lib INTERFACE src)

# The dg_lib library needs to know where to find Eigen's headers.
# Using target_link_libraries with an INTERFACE library like Eigen3::Eigen
# handles both include directories and other usage requirements.
target_link_libraries(dg_lib INTERFACE Eigen3::Eigen)


# --- Executables ---
# Create executables from your main .cpp files.
# add_executable(DG DG.cpp)
# add_executable(DGSEM DGSEM.cpp)

# Link the executables against your project's library.
# This will also link them with Eigen because dg_lib has a PUBLIC dependency on it.
# target_link_libraries(DG PRIVATE dg_lib)
# target_link_libraries(DGSEM PRIVATE dg_lib)

# --- Installation (Optional) ---
# This part is optional, but it's good practice. It allows you to install
# your executables to a system-wide location (e.g., /usr/local/bin)
# by running 'make install' after building.
# install(TARGETS DG DGSEM
    # RUNTIME DESTINATION bin
# )

