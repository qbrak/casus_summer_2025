import minterpy as mp
import numpy as np
import time

def M(x):
    """
    Example of input values:
        x = [[1,2,3,4,5,6,7,8], 
            [1,2,3,4,5,6,7,8],
            [1,2,3,4,5,6,7,8],
            ....
            [1,2,3,4,5,6,7,8]]
    
    """
    r_w = x[:, 0]
    r = x[:, 1]
    T_u = x[:, 2]
    H_u = x[:, 3]
    T_l = x[:, 4]
    H_l = x[:, 5]
    L = x[:, 6]
    K_w = x[:, 7]

    return 2 * np.pi * T_u * (H_u - H_l) / (np.log(r / r_w) * (1 + T_u / T_l) + 2 * L *T_u / ((r_w**2)*K_w))


def MU(x):
    """
        Here we perform the scaling from [-1,1] to the original bounds
        of the input variables.
    
    """
    bounds = [
       [0.15, 0.05], # r_w
       [50_000, 100], # r
       [115_600, 63_070], # T_u
       [1_100, 990], # H_u
       [116, 63.1], # T_l
       [820, 700], # H_l
       [1_680, 1120], # L
       [12_045, 9_985], # K_w
    ]
    abounds = np.array(bounds)
    # breakpoint()
    mu = 0.5 * (abounds[:,1] + abounds[:,0])
    sigma = 0.5 * (abounds[:,1] - abounds[:,0])

    # Normalize the input
    x = x * sigma + mu

    return M(x)


def interpolate(m, n, p, fun):
    """
        Interpolates the function fun using the Chebyshev nodes.
        Returns a polynomial in Newton form.
    """

    mis = mp.MultiIndexSet.from_degree(m, n, p)
    grid = mp.Grid(mis)
    nodes = grid.unisolvent_nodes

    Lagrange_poly = mp.LagrangePolynomial.from_grid(grid, fun(nodes))

    return mp.LagrangeToNewton(Lagrange_poly)()


def RMSE(m, poly, fun, N=1000):
    # Generate N random points from m-dimensional cube [-1,1]^m
    x = np.random.uniform(-1, 1, (N, m))
    
    # Evaluate polynomial and function at these points
    poly_vals = poly(x)
    fun_vals = fun(x)
    
    # Compute root mean square error
    rmse = np.sqrt(np.mean((poly_vals - fun_vals)**2))
    
    return rmse


def main():
    m = 8
    n = 5
    p = 2
    
    # Time the interpolation
    start_time = time.time()
    poly = interpolate(m, n, p, MU)
    interpolate_time = time.time() - start_time
    print(f"Time spent in interpolate: {interpolate_time:.4f} seconds")

    # Time the RMSE calculation
    start_time = time.time()
    rmse = RMSE(m, poly, MU)
    rmse_time = time.time() - start_time
    print(f"Time spent in RMSE: {rmse_time:.4f} seconds")
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()
