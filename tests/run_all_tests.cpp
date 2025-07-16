#include "test_framework.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

// Test runner that compiles and runs all individual test files
int main() {
    std::cout << "=== Discontinuous Galerkin Framework Test Suite ===\n";
    std::cout << "Running comprehensive tests for all components...\n\n";
    
    // List of test files to run
    std::vector<std::string> test_files = {
        "test_euler",
        "test_legendre", 
        "test_gauss_legendre",
        "test_field",
        "test_rusanov",
        "test_dg_scheme",
        "test_runge_kutta",
        "test_lagrange",
        "test_mesh"
    };
    
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    
    for (const auto& test_name : test_files) {
        std::cout << std::string(60, '=') << "\n";
        std::cout << "Running " << test_name << "...\n";
        std::cout << std::string(60, '=') << "\n";
        
        // Check if we're on Windows or Linux/WSL
        std::string command;
#ifdef _WIN32
        command = "./" + test_name + ".exe";
#else
        command = "./" + test_name;
#endif
        
        int result = std::system(command.c_str());
        
        if (result == 0) {
            std::cout << "✓ " << test_name << " PASSED\n";
            passed_tests++;
        } else {
            std::cout << "✗ " << test_name << " FAILED\n";
            failed_tests++;
        }
        total_tests++;
        std::cout << "\n";
    }
    
    std::cout << std::string(60, '=') << "\n";
    std::cout << "FINAL RESULTS:\n";
    std::cout << "Total test files: " << total_tests << "\n";
    std::cout << "Passed: " << passed_tests << "\n";
    std::cout << "Failed: " << failed_tests << "\n";
    
    if (failed_tests == 0) {
        std::cout << "🎉 ALL TESTS PASSED! 🎉\n";
        return 0;
    } else {
        std::cout << "❌ " << failed_tests << " test file(s) failed\n";
        return 1;
    }
}
