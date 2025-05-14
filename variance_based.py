import numpy as np
import minterpy as mp
from scipy.special import eval_legendre
from scipy.linalg import solve
import time
from tabulate import tabulate

from main import MU, interpolate, input_names


############## LEGENDRE CONVERSION ####################

def eval_legendre_multidim(exponents, nodes):
    r"""
        Evaluates the Legendre polynomial at the given nodes.

        We need to normalize the scipy polynomial by sqrt(2*n + 1) to get the correct normalization.

        This is because we define the Legendre polynomial scalar product as:
        $$ \langle P_m, P_n \rangle = \int_{-1}^1 f(x) g(x) \frac{1}{2}dx $$
    """
    values = np.ones(len(nodes))
    for dim, n in enumerate(exponents):
        normalization_factor = np.sqrt( (2*n + 1) )
        x = nodes[:, dim]
        values *= normalization_factor * eval_legendre(n, x)
    return values
   

def convert_to_legendre(grid, unisolvent_node_vals):
    """
        Interpolates the grid with values unisolvent_node_vals in the Legendre basis.

        Returns the coefficients in the Legendre basis in the order (for exponents)
        specified in the grid.

        This function constructs the Vandermonde matrix from the Legendre to the evaluated points:

        $$
            V * C_{leg} = f(nodes)
        $$

        where $VA$ is the Legendre polynomials evaluated at the nodes.
        The f(nodes) are the unisolvent_node_values.
    """

    nodes = grid.unisolvent_nodes
    exponents = grid.multi_index.exponents

    VA = np.zeros((len(nodes), len(nodes)))
    for j in range(len(nodes)):
        VA[:, j] = eval_legendre_multidim(exponents[j,:], nodes) # we need to pass a 2D array to the polynomial

    leg_coeffs = solve(VA, unisolvent_node_vals)
    return leg_coeffs

############## MONTE CARLO ####################

def mean_mc(m, fun, N=1000):
    """
        Compute the mean of the function using Monte Carlo method.
    """
    x = np.random.uniform(-1, 1, (N, m))
    fun_vals = fun(x)
    mean = np.mean(fun_vals)
    
    return mean


def var_mc(m, fun, N=1000):
    """
        Compute the variance of the function using Monte Carlo method.
    """
    x = np.random.uniform(-1, 1, (N, m))
    fun_vals = fun(x) 
    var = np.var(fun_vals)
    
    return var

def sobol_effect(input, m, fun, N=1000):
    r"""
        Compute the Sobol's total and main effect of the input on the variance of the polynomial.
        Using Monte Carlo simulation.

        We compute both at the same time, to reuse resources.

        The main effect is defined as:
            $$ S_j \approx \frac{1}{N} \sum_{i=1}^N f(x_j^{(i)}, \mathbf{x}_{\{\sim j\}}^{(i)}) f(x_j^{(i)}, \mathbf{z}_{\{\sim j\}}^{(i)})  - \left(\frac{1}{N}\sum_{i=1}^N f(x^{(i)})\right)^2$$

        and the total effect is defined as:
            $$ ST_j \approx \frac{1}{2N} \sum_{i=1}^N\left(f(x_j^{(i)}, \mathbf{x}_{\{\sim j\}}^{(i)}) - f(z_j^{(i)}, \mathbf{x}_{\{\sim j\}}^{(i)}) \right)^2 $$
        
        where $\{mathbf{x}_j\}$ and $\{mathbf{z}_j\}$ are independently generated points.

        Returns:
            * S_input: the Sobol's main effect of the input on the variance of the polynomial.
            * ST_input: the Sobol's total effect of the input on the variance of the polynomial.
    """
    x = np.random.uniform(-1, 1, (N, m))
    z = np.random.uniform(-1, 1, (N, m))
    zmod1 = x.copy()
    zmod1[:, input] = z[:, input]

    zmodn_1 = z
    zmodn_1[:, input] = x[:, input]

    var = var_mc(m, fun, N)
    mean = mean_mc(m, fun, N)

    fun_vals = fun(x)

    S_input = (np.mean(fun_vals*fun(zmodn_1)) - mean**2) / var
    ST_input =  0.5 * np.mean((fun_vals - fun(zmod1))**2) / var
    return S_input, ST_input 

    

############### LEGENDRE POLYNOMIALS ####################

def mean_leg(leg_coeffs):
    """
        Compute the mean of the function based on the Legendre coefficients.
    """
    return leg_coeffs[0] 

def var_leg(leg_coeffs):
    """
        Compute the variance of the function based on the Legendre coefficients.
    """
    return np.sum(leg_coeffs[1:]**2)

def _var_masked(leg_coeffs, mask):
    """
        Compute the variance of the function based on the Legendre coefficients.
        We use a mask to select the coefficients we want to include in the variance.

        We always skip the first coefficient, which is the mean of the polynomial.
    """
    mask[0] = False # we don't want to include the mean in the variance
    if not mask.any():
        return np.nan
    masked_coeffs = leg_coeffs[mask]
    var_sum = np.sum(masked_coeffs**2)
    return var_sum / var_leg(leg_coeffs)

def sobol_total_effect_leg(input, mis, leg_coeffs):
    r"""
        Compute the Sobol's total effect of the input on the variance of the polynomial.

        The total effect is defined as:
            $$ ST_i = \frac{V_i+V_{i,j}+V_{i,j,k}+\dots+V_{i,\dots,m}}{Var(V)} $$
        
        where $V_i$ is the variance of the polynomial with respect to the input $i$ defined as:
            * $V_i = \mathbb{V}_{\sim X_i}[\mathbb{E}[Y|X_i=x_i]]$
            * $V_{i,j} = \mathbb{V}_{\sim X_i,X_j}[\mathbb{E}[Y|X_i=x_i,X_j=x_j]] - V_i - V_j$
            * etc...
        
        In the Legendre basis this becomes:
            $$ ST_i = \frac{1}{\mathbb{V}[Y]} \sum_{\alpha \in B_i}c_{\alpha}^2; \; B_i=\{\alpha \in \mathbb{N}^m: \alpha_i > 0\} $$
        where $c_{\alpha}$ are the coefficients of the polynomial in Legendre basis.
    """
    mask = mis.exponents[:, input] > 0 
    return _var_masked(leg_coeffs, mask)

def sobol_main_effect_leg(input, mis, leg_coeffs):
    r"""
        Compute the Sobol's main effect of the input on the variance of the polynomial.

        The main effect is defined as:
            $$ S_i = \frac{V_i}{Var(V)} $$
        
        where $V_i$ is the variance of the polynomial with respect to the input $i$ defined as:
            * $V_i = \mathbb{V}_{\sim X_i}[\mathbb{E}[Y|X_i=x_i]]$
        
        In the Legendre basis this becomes:
            $$ S_i = \frac{1}{\mathbb{V}[Y]} \sum_{\alpha \in B_i}c_{\alpha}^2; \; B_i=\{\alpha \in \mathbb{N}^m: \alpha_i > 0\} $$
        where $c_{\alpha}$ are the coefficients of the polynomial in Legendre basis.
    """
    mask = mis.exponents[:, input] > 0
    dims = mis.exponents.shape[1]
    for i in range(dims):
        if i != input:
            mask &= mis.exponents[:, i] == 0

    return _var_masked(leg_coeffs, mask)


def effective_dimension_superposition(mis, leg_coeffs, p=0.99):
    r"""
        Compute the effective dimension of the polynomial based on the Legendre coefficients.
        The effective dimension is defined as the number of inputs that contribute to the variance of the polynomial.
        We compute this by summing the squares of the coefficients and checking if they exceed p.

        The formula for $m_s$ is:
        $$
            m_s = \min\{m: \sum_{|u|\leq m} \sigma_{u}^2 \geq p \sigma^2 \}
        $$

        Where $|u|$ is the number of non-zero elements in the vector $u$.
    """
    dims = mis.exponents.shape[1]

    m = np.nan
    for m in range(1, dims):
        mask = np.sum(mis.exponents > 0, axis=1) <= m
        u = _var_masked(leg_coeffs, mask)
        if u > p:
            break
    return m

def effective_dimension_truncation(mis, leg_coeffs, order, p=0.99):
    r"""
        Compute the effective dimension of the polynomial based on the Legendre coefficients.

        The formula for $m_t$ is:
        $$
            m_t = \min\{m: \sum_{u \subseteq [1:m]} \sigma_{u}^2 \geq p \sigma^2 \}
        $$
        
        In the formula above we assume that the coefficients are ordered with Sobol's
        total order. This is provided to the function with order input0.
    """
    dims = mis.exponents.shape[1]
    exponents = mis.exponents[:, order]

    m = np.nan
    for m in range(0, dims):
        mask = (exponents[:, m+1:] == 0).all(axis=1)
        u = _var_masked(leg_coeffs, mask)
        if u > p:
            m = m + 1 # we need to add 1 because we are using 0-based indexing
            break
    return m


def main():
    
    m = 8
    
    # The interpolation approach
    n = 7
    p = 1
  
    mis = mp.MultiIndexSet.from_degree(m, n, p)
    grid = mp.Grid(mis)
    nodes = grid.unisolvent_nodes
    unisolvent_node_vals = MU(nodes)

    start_time = time.time()
    leg_coeffs = convert_to_legendre(grid, unisolvent_node_vals)
    legendre_time = time.time() - start_time
    
    start_time = time.time()
    total_imp_leg = [
        (i, sobol_total_effect_leg(i, mis, leg_coeffs))
        for i in range(m)
    ]
    main_imp_leg = [
        (i, sobol_main_effect_leg(i, mis, leg_coeffs))
        for i in range(m)
    ]
    sobol_time = time.time() - start_time
     
    sorted_total_imp_leg = sorted(total_imp_leg, key=lambda x: x[1], reverse=True)
    sorted_main_imp_leg = sorted(main_imp_leg, key=lambda x: x[1], reverse=True)
     
    # The Monte Carlo approach
    N = int((m+2)*1e5)

    start_time = time.time()
    fun_mean = mean_mc(m, MU, N=N)
    fun_var = var_mc(m, MU, N=N)

    importances_mc = dict(
        (i, sobol_effect(i, m, MU, N=N))
        for i in range(m)
    )
    monte_carlo_time = time.time() - start_time

    print(f"Using {N} points for Monte Carlo")
    print(f"Using p={p}, n={n} for interpolation")
    print(f"\nMean difference rel ((monte_carlo - interpolation) / monte_carlo):")
    print(f"{abs(fun_mean - mean_leg(leg_coeffs)) / fun_mean:.6f} = {fun_mean - mean_leg(leg_coeffs):.6f} / {fun_mean:.6f}")
    print(f"Var difference rel ((monte_carlo - interpolation) / monte_carlo):")
    print(f"{abs(fun_var - var_leg(leg_coeffs)) / fun_var:.6f} = {fun_var - var_leg(leg_coeffs):.6f} / {fun_var:.6f}")

    print("\nSobol's main effect")
    # Create table headers and data
    headers = ["Input", "Legendre", "Monte Carlo", "Abs Diff"]
    table_data = []
    
    # Add the Sobol's main effect to the table
    for i, st in sorted_main_imp_leg:
        mc = importances_mc[i][0]
        table_data.append([f"{i}: ${input_names[i]}$", f"{st:.6f}", f"{mc:.6f}", f"{abs(st-mc):.6f}"])
    
    # Print the table using tabulate with latex format
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\nSobol's total effect")
    # Create table headers and data
    headers = ["Input", "Legendre", "Monte Carlo", "Abs Diff"]
    table_data = []
    
    for i, st in sorted_total_imp_leg:
        mc = importances_mc[i][1] # Monte Carlo total effect
        table_data.append([f"{i}: ${input_names[i]}$", f"{st:.6f}", f"{mc:.6f}", f"{abs(st-mc):.6f}"])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\nEffective dimension:")
    print(f"   Superposition: {effective_dimension_superposition(mis, leg_coeffs):.6f}")
    order = np.argsort(-np.array([x[1] for x in total_imp_leg]))
    print(f"   Truncation: {effective_dimension_truncation(mis, leg_coeffs, order):.6f}")


    print(f"\nTiming:")
    print(f"Convert to Legendre: {legendre_time:.4f} seconds")
    print(f"Sobol calculations: {sobol_time:.4f} seconds")
    print(f"Monte Carlo calculations: {monte_carlo_time:.4f} seconds")
    print(f"Total: {legendre_time + sobol_time + monte_carlo_time:.4f} seconds")

if __name__ == "__main__":
    main()
