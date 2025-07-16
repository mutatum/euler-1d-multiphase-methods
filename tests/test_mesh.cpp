#include "test_framework.hpp"
#include "../src/mesh/mesh.hpp"
#include "../src/mesh/mesh_1d.hpp"

TestFramework global_test_framework;

// Test base Mesh class
TEST(base_mesh_construction) {
    Mesh mesh;
    ASSERT_EQ(0, mesh.total_cells());
    
    Mesh mesh_with_cells(10);
    ASSERT_EQ(10, mesh_with_cells.total_cells());
}

// Test mesh_structured_uniform_1d construction
TEST(mesh_1d_construction) {
    mesh_structured_uniform_1d mesh(0.0, 1.0, 10, 2);
    
    ASSERT_EQ(10, mesh.num_cells());
    ASSERT_EQ(2, mesh.num_ghost_cells());
    ASSERT_EQ(14, mesh.total_cells()); // 10 real + 4 ghost
    ASSERT_EQ(0.0, mesh.domain_start());
    ASSERT_EQ(1.0, mesh.domain_end());
    ASSERT_EQ(0.1, mesh.dx());
}

// Test mesh_1d construction with invalid parameters
TEST(mesh_1d_invalid_construction) {
    // Test invalid domain
    ASSERT_THROWS(mesh_structured_uniform_1d(1.0, 0.0, 10, 2));
    
    // Test zero cells
    ASSERT_THROWS(mesh_structured_uniform_1d(0.0, 1.0, 0, 2));
    
    // Test zero ghost cells
    ASSERT_THROWS(mesh_structured_uniform_1d(0.0, 1.0, 10, 0));
}

// Test cell centers
TEST(mesh_1d_cell_centers) {
    mesh_structured_uniform_1d mesh(0.0, 1.0, 4, 1);
    
    // Expected cell centers for 4 cells in [0,1] with 1 ghost cell:
    // Ghost: -0.125, Real: 0.125, 0.375, 0.625, 0.875
    
    ASSERT_NEAR(-0.125, mesh.cell_center(0), 1e-14); // Ghost cell
    ASSERT_NEAR(0.125, mesh.cell_center(1), 1e-14);  // First real cell
    ASSERT_NEAR(0.375, mesh.cell_center(2), 1e-14);  // Second real cell
    ASSERT_NEAR(0.625, mesh.cell_center(3), 1e-14);  // Third real cell
    ASSERT_NEAR(0.875, mesh.cell_center(4), 1e-14);  // Fourth real cell
}

// Test mesh with negative domain
TEST(mesh_1d_negative_domain) {
    mesh_structured_uniform_1d mesh(-2.0, -1.0, 5, 1);
    
    ASSERT_EQ(5, mesh.num_cells());
    ASSERT_EQ(-2.0, mesh.domain_start());
    ASSERT_EQ(-1.0, mesh.domain_end());
    ASSERT_EQ(0.2, mesh.dx());
    
    // Test cell centers
    ASSERT_NEAR(-2.3, mesh.cell_center(0), 1e-14); // Ghost cell
    ASSERT_NEAR(-1.9, mesh.cell_center(1), 1e-14); // First real cell
    ASSERT_NEAR(-1.1, mesh.cell_center(5), 1e-14); // Last real cell
}

// Test mesh with symmetric domain
TEST(mesh_1d_symmetric_domain) {
    mesh_structured_uniform_1d mesh(-1.0, 1.0, 4, 2);
    
    ASSERT_EQ(4, mesh.num_cells());
    ASSERT_EQ(-1.0, mesh.domain_start());
    ASSERT_EQ(1.0, mesh.domain_end());
    ASSERT_EQ(0.5, mesh.dx());
    
    // Test cell centers
    ASSERT_NEAR(-1.75, mesh.cell_center(0), 1e-14); // First ghost cell
    ASSERT_NEAR(-1.25, mesh.cell_center(1), 1e-14); // Second ghost cell
    ASSERT_NEAR(-0.75, mesh.cell_center(2), 1e-14); // First real cell
    ASSERT_NEAR(-0.25, mesh.cell_center(3), 1e-14); // Second real cell
    ASSERT_NEAR(0.25, mesh.cell_center(4), 1e-14);  // Third real cell
    ASSERT_NEAR(0.75, mesh.cell_center(5), 1e-14);  // Fourth real cell
}

// Test mesh with large number of cells
TEST(mesh_1d_large_mesh) {
    mesh_structured_uniform_1d mesh(0.0, 10.0, 1000, 3);
    
    ASSERT_EQ(1000, mesh.num_cells());
    ASSERT_EQ(3, mesh.num_ghost_cells());
    ASSERT_EQ(1006, mesh.total_cells()); // 1000 real + 6 ghost
    ASSERT_EQ(0.01, mesh.dx());
    
    // Test some cell centers
    ASSERT_NEAR(-0.025, mesh.cell_center(0), 1e-14); // First ghost cell
    ASSERT_NEAR(0.005, mesh.cell_center(3), 1e-14);  // First real cell
    ASSERT_NEAR(9.995, mesh.cell_center(1002), 1e-14); // Last real cell
}

// Test mesh with small domain
TEST(mesh_1d_small_domain) {
    mesh_structured_uniform_1d mesh(0.0, 1e-6, 10, 1);
    
    ASSERT_EQ(10, mesh.num_cells());
    ASSERT_EQ(1e-6, mesh.domain_end());
    ASSERT_EQ(1e-7, mesh.dx());
    
    // Test cell centers
    ASSERT_NEAR(-0.5e-7, mesh.cell_center(0), 1e-21); // Ghost cell
    ASSERT_NEAR(0.5e-7, mesh.cell_center(1), 1e-21);  // First real cell
}

// Test edge cases for cell center access
TEST(mesh_1d_cell_center_bounds) {
    mesh_structured_uniform_1d mesh(0.0, 1.0, 5, 2);
    
    // Test valid indices
    ASSERT_TRUE(std::isfinite(mesh.cell_center(0)));
    ASSERT_TRUE(std::isfinite(mesh.cell_center(mesh.total_cells() - 1)));
    
    // Test boundary values
    ASSERT_THROWS(mesh.cell_center(mesh.total_cells()));
    ASSERT_THROWS(mesh.cell_center(100));
}

// Test mesh consistency
TEST(mesh_1d_consistency) {
    mesh_structured_uniform_1d mesh(0.0, 1.0, 8, 2);
    
    // Test that dx is consistent with domain and number of cells
    double expected_dx = (mesh.domain_end() - mesh.domain_start()) / mesh.num_cells();
    ASSERT_NEAR(expected_dx, mesh.dx(), 1e-14);
    
    // Test that cell centers are properly spaced
    for (std::size_t i = 1; i < mesh.total_cells(); ++i) {
        double spacing = mesh.cell_center(i) - mesh.cell_center(i-1);
        ASSERT_NEAR(mesh.dx(), spacing, 1e-14);
    }
}

int main() {
    std::cout << "Running mesh tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
