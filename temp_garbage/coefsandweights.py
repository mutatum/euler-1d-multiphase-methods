# %%
import numpy as np
from numpy.polynomial.legendre import leggauss, Legendre

def gauss_legendre(N):
    """
    Gauss–Legendre quadrature of order N.
    Returns nodes x_i and weights w_i on [-1,1].
    """
    x, w = leggauss(N)
    return x, w


def gauss_lobatto_legendre(N):
    """
    Gauss–Lobatto–Legendre quadrature with N nodes.
    Nodes include -1 and +1, with N-2 interior roots of P'_{N-1}(x)=0.
    Weights: w_i = 2/[N(N-1) * (P_{N-1}(x_i))^2], endpoints w = 2/[N(N-1)].
    """
    if N < 2:
        raise ValueError("Need N>=2 for GLL.")
    # 1) Get P_{N-1}
    P = Legendre.basis(N-1)
    # 2) find interior roots of P'
    dP = P.deriv()
    xi_int = np.sort(dP.roots())
    # 3) assemble full node array
    x = np.empty(N)
    x[0], x[-1] = -1.0, 1.0
    x[1:-1] = xi_int
    # 4) compute weights
    w = np.empty_like(x)
    w[0] = w[-1] = 2.0/(N*(N-1))
    Pnm1 = P(x)           # P_{N-1}(x_i)
    w[1:-1] = 2.0/(N*(N-1)*(Pnm1[1:-1]**2))
    return x, w

def generate_gl_cpp_header(max_n, filename="gl_quadrature_data.hpp"):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write("namespace GL {\n")
        f.write("template<typename Scalar, std::size_t N> struct GLData;\n\n")

        for n in range(1, max_n + 1):
            x, w = gauss_legendre(n)
            f.write(f"template<typename Scalar>\n")
            f.write(f"struct GLData<Scalar, {n}> {{\n")
            
            nodes_str = ", ".join([f"{val:.18f}" for val in x])
            f.write(f"    static constexpr std::array<Scalar, {n}> nodes = {{{nodes_str}}};\n")
            
            weights_str = ", ".join([f"{val:.18f}" for val in w])
            f.write(f"    static constexpr std::array<Scalar, {n}> weights = {{{weights_str}}};\n")
            
            f.write("};\n\n")
        
        f.write("} // namespace GL\n")

def generate_cpp_header(max_n, filename="gll_quadrature_data.hpp"):
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <array>\n\n")
        f.write("namespace GLL {\n")
        f.write("template<typename Scalar, std::size_t N> struct GLLData;\n\n")

        for n in range(2, max_n + 1):
            x, w = gauss_lobatto_legendre(n)
            f.write(f"template<typename Scalar>\n")
            f.write(f"struct GLLData<Scalar, {n}> {{\n")
            
            nodes_str = ", ".join([f"{val:.18f}" for val in x])
            f.write(f"    static constexpr std::array<Scalar, {n}> nodes = {{{nodes_str}}};\n")
            
            weights_str = ", ".join([f"{val:.18f}" for val in w])
            f.write(f"    static constexpr std::array<Scalar, {n}> weights = {{{weights_str}}};\n")
            
            f.write("};\n\n")
        
        f.write("} // namespace GLL\n")


# Example usage
if __name__ == "__main__":
    generate_cpp_header(15, "src/utils/quadrature/gll_quadrature_data.hpp")
    print("Generated gll_quadrature_data.hpp")
    generate_gl_cpp_header(15, "src/utils/quadrature/gl_quadrature_data.hpp")
    print("Generated gl_quadrature_data.hpp")
    # x_gl, w_gl = gauss_legendre(12)
    # x_gll, w_gll = gauss_lobatto_legendre(12)
    # print("GL nodes:", x_gl)
    # print("GL weights:", w_gl)
    # print("GLL nodes:", x_gll)
    # print("GLL weights:", w_gll)

