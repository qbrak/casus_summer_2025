import minterpy as mp
import numpy as np
import time
from tabulate import tabulate

bounds = np.array([
    [0.15, 0.05], # r_w
    [50_000, 100], # r
    [115_600, 63_070], # T_u
    [1_100, 990], # H_u
    [116, 63.1], # T_l
    [820, 700], # H_l
    [1_680, 1120], # L
    [12_045, 9_985], # K_w
])

input_names = [
    "r_w",
    "r",
    "T_u",
    "H_u",
    "T_l",
    "H_l",
    "L",
    "K_w",
]

mu = 0.5 * (bounds[:,1] + bounds[:,0])
sigma = 0.5 * (bounds[:,1] - bounds[:,0])

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

    mu = 0.5 * (bounds[:,1] + bounds[:,0])

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

    unisolvent_node_vals = fun(nodes)
    Lagrange_poly = mp.LagrangePolynomial.from_grid(grid, fun(nodes))

    return Lagrange_poly, unisolvent_node_vals


def RMSE(m, poly, fun, N=1000):
    # Generate N random points from m-dimensional cube [-1,1]^m
    x = np.random.uniform(-1, 1, (N, m))
    
    # Evaluate polynomial and function at these points
    poly_vals = poly(x)
    fun_vals = fun(x)
    
    # Compute root mean square error
    rmse = np.sqrt(np.mean((poly_vals - fun_vals)**2))
    rmse_rel = rmse / np.std(fun_vals)
    
    return rmse, rmse_rel


def main():
    m = 8

    pairs = (
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7), # Used in variance-based sensitivity analysis
        (2, 5),
    )

    # Create lists to store results
    results = []
    
    for p, n in pairs:
        print(f"Running interpolation with p: {p}, n: {n}")
        
        # Time the interpolation
        start_time = time.time()
        poly, unisolvent_node_vals = interpolate(m, n, p, MU)
        interpolate_time = time.time() - start_time
        
        # Calculate RMSE (without timing it separately)
        poly_newton = mp.LagrangeToNewton(poly)()
        _, rmse_rel = RMSE(m, poly_newton, MU)
        
        # Store the results
        results.append([p, n, rmse_rel,
                        len(unisolvent_node_vals), 
                        interpolate_time, ])
    
    # Display results in a table
    headers = ["$p$", "$n$", "Rel RMSE", "Node count", 
               "Interp Time (s)",]
    
    print("\nResults Table:")
    print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=".6f"))




if __name__ == "__main__":
    main()
