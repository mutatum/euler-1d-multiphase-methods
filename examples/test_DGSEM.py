import unittest
import numpy as np
import colorama
from colorama import Fore, Style, Back
from DGSEM import lagrange_basis, GLL_quadrature, eq_of_state, physical_flux, rusanov, compute_differentiation_matrix, init_sod, compute_max_speed

colorama.init(autoreset=True)

class PrettyTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.writeln(Fore.GREEN + Back.GREEN + " PASS " + Style.RESET_ALL + f" {test}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.writeln(Fore.RED + Back.RED + " FAIL " + Style.RESET_ALL + f" {test}")

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.writeln(Fore.YELLOW + Back.YELLOW + " ERROR " + Style.RESET_ALL + f" {test}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.stream.writeln(Fore.CYAN + Back.CYAN + " SKIP " + Style.RESET_ALL + f" {test}")

class PrettyTestRunner(unittest.TextTestRunner):
    resultclass = PrettyTestResult

class TestDGSEM(unittest.TestCase):

    def test_lagrange_basis(self):
        p = 2
        xi = GLL_quadrature[p]['xi']
        
        # Test for i = 0
        self.assertAlmostEqual(lagrange_basis(-1.0, xi, 0), 1.0)
        self.assertAlmostEqual(lagrange_basis(0.0, xi, 0), 0.0)
        self.assertAlmostEqual(lagrange_basis(1.0, xi, 0), 0.0)

        # Test for i = 1
        self.assertAlmostEqual(lagrange_basis(-1.0, xi, 1), 0.0)
        self.assertAlmostEqual(lagrange_basis(0.0, xi, 1), 1.0)
        self.assertAlmostEqual(lagrange_basis(1.0, xi, 1), 0.0)

        # Test for i = 2
        self.assertAlmostEqual(lagrange_basis(-1.0, xi, 2), 0.0)
        self.assertAlmostEqual(lagrange_basis(0.0, xi, 2), 0.0)
        self.assertAlmostEqual(lagrange_basis(1.0, xi, 2), 1.0)

        # Test for a value in between
        xi_eval = 0.5
        self.assertAlmostEqual(lagrange_basis(xi_eval, xi, 0), -0.125)
        self.assertAlmostEqual(lagrange_basis(xi_eval, xi, 1), 0.75)
        self.assertAlmostEqual(lagrange_basis(xi_eval, xi, 2), 0.375)

    def test_eq_of_state(self):
        gamma = 1.4
        rho = 1.0
        e = 2.5
        self.assertAlmostEqual(eq_of_state(gamma, rho, e), 1.0)

    def test_physical_flux(self):
        rho = 1.0
        u = 2.0
        p = 3.0
        gamma = 1.4
        e = p / ((gamma - 1) * rho)
        E = e + 0.5 * u**2
        U = np.array([rho, rho*u, rho*E])
        
        flux = physical_flux(U)
        
        expected_flux = np.array([
            rho * u,
            rho * u**2 + p,
            (rho * E + p) * u
        ])
        
        np.testing.assert_allclose(flux, expected_flux, rtol=1e-6)

    def test_rusanov(self):
        Ul = np.array([1.0, 0.0, 2.5]) # rho=1, u=0, p=1
        Ur = np.array([0.125, 0.0, 2.0]) # rho=0.125, u=0, p=0.1
        smax = 1.0
        
        flux_l = physical_flux(Ul)
        flux_r = physical_flux(Ur)
        
        expected_rusanov = 0.5 * (flux_l + flux_r) - 0.5 * smax * (Ur - Ul)
        
        rusanov_flux = rusanov(Ul, Ur, smax)
        np.testing.assert_allclose(rusanov_flux, expected_rusanov)

    def test_compute_differentiation_matrix(self):
        p = 2
        D = compute_differentiation_matrix(p)
        
        # Check that the rows sum to zero
        self.assertTrue(np.allclose(np.sum(D, axis=1), 0.0))

    def test_init_sod(self):
        N = 100
        p = 3
        U_cells = np.zeros((3, N, p + 1))
        Omega = [0, 1]
        X = np.linspace(*Omega, N + 1)
        cell_centers = (X[1:] + X[:-1]) / 2.0
        dx = (Omega[1] - Omega[0]) / N
        
        init_sod(U_cells, cell_centers, dx)
        
        # Check a point in the left state (x < 0.5)
        x_nodes = cell_centers[24] + 0.5 * dx * GLL_quadrature[p]['xi']
        self.assertTrue(np.all(x_nodes < 0.5))
        rho = U_cells[0, 24, :]
        u = U_cells[1, 24, :] / rho
        E = U_cells[2, 24, :] / rho
        p_val = (1.4-1)*rho*(E - 0.5*u**2)
        
        self.assertTrue(np.allclose(rho, 1.0))
        self.assertTrue(np.allclose(u, 0.0))
        self.assertTrue(np.allclose(p_val, 1.0))

        # Check a point in the right state (x > 0.5)
        x_nodes = cell_centers[74] + 0.5 * dx * GLL_quadrature[p]['xi']
        self.assertTrue(np.all(x_nodes > 0.5))
        rho = U_cells[0, 74, :]
        u = U_cells[1, 74, :] / rho
        E = U_cells[2, 74, :] / rho
        p_val = (1.4-1)*rho*(E - 0.5*u**2)
        
        self.assertTrue(np.allclose(rho, 0.125))
        self.assertTrue(np.allclose(u, 0.0))
        self.assertTrue(np.allclose(p_val, 0.1))
        
        # Check the discontinuity at cell 49/50
        x_nodes = cell_centers[49] + 0.5 * dx * GLL_quadrature[p]['xi']
        for j in range(p+1):
            rho = U_cells[0, 49, j]
            if x_nodes[j] < 0.5:
                self.assertAlmostEqual(rho, 1.0)
            else:
                self.assertAlmostEqual(rho, 0.125)

    def test_compute_max_speed(self):
        N = 10
        p = 3
        U_cells = np.zeros((3, N, p + 1))
        Omega = [0, 1]
        X = np.linspace(*Omega, N + 1)
        cell_centers = (X[1:] + X[:-1]) / 2.0
        dx = (Omega[1] - Omega[0]) / N
        init_sod(U_cells, cell_centers, dx)
        
        lambda_max = compute_max_speed(U_cells)
        self.assertGreater(lambda_max, 0.0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDGSEM)
    PrettyTestRunner(verbosity=2).run(suite)
