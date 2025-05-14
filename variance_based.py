import numpy as np
import minterpy as mp
from scipy.special import eval_legendre
from scipy.linalg import solve
import time
from tabulate import tabulate


from main import MU, interpolate

def var(m, poly):
    r"""
        Compute the variance of the polynomial, assume it is in Newton form (to square it)

        Assuming:
           $$ Z = Q(X_1, X_2, \ldots, X_m) $$
        
        where $X_i$ are independent and uniformly distributed in [-1, 1].

        We will use the formula that:
              $$ 
              Var(Z) = E[Z^2] - (E[Z])^2 = 
              \int_{\square^m}Q(\mathbf{X})^2 d\mathbf{X} - \left(\int_{\square^m}Q(\mathbf{X}) d\mathbf{X}\right)
              $$
          
          where $E[Z]$ is the expected value of $Z$.

          We do this by using the minterpy library to compute the integral of the polynomial.
    """

    poly2 = poly * poly # FIXME: takes way too much time
    scaling = 1 / (2**m) # This is because the density in each dimension is 1/2 * lebesgue measure
    return poly2.integrate_over()*scaling - (poly.integrate_over()*scaling)**2

def eval_legendre_multidim(exponents, nodes):
    r"""
        Evaluates the Legendre polynomial at the given nodes.

        We need to normalize the scipy polynomial by 1/(2*n + 1) to get the correct normalization.

        This is because we define the Legendre polynomial scalar product as:
        $$ \langle P_m, P_n \rangle = \int_{-1}^1 f(x) g(x) \frac{1}{2}dx $$
    """
    values = np.ones(len(nodes))
    for dim, n in enumerate(exponents):
        normalization_factor = np.sqrt( (2*n + 1) )
        x = nodes[:, dim]
        values *= normalization_factor * eval_legendre(n, x)
    return values
   

def convert_to_legendre(poly, unisolvent_node_vals):
    """
    Converts poly from Lagrange to Legendre basis.

    Returns the MultiIndexSet and the coefficients of the polynomial in Legendre basis.
    """

    mis = poly.multi_index
    grid = poly.grid
    nodes = grid.unisolvent_nodes

    VA = np.zeros((len(nodes), len(nodes)))
    exponents = mis.exponents
    for j in range(len(nodes)):
        VA[:, j] = eval_legendre_multidim(exponents[j,:], nodes) # we need to pass a 2D array to the polynomial

    leg_coeffs = solve(VA, unisolvent_node_vals)
    return mis, leg_coeffs

############## MONTE CARLO ####################

def mean_mc(m, fun, N=1000):
    """
        Compute the mean of the function using Monte Carlo method.
    """
    # Generate N random points from m-dimensional cube [-1,1]^m
    x = np.random.uniform(-1, 1, (N, m))
    
    # Evaluate function at these points
    fun_vals = fun(x)
    
    # Compute mean
    mean = np.mean(fun_vals)
    
    return mean


def var_mc(m, fun, N=1000):
    """
        Compute the variance of the function using Monte Carlo method.
    """
    # Generate N random points from m-dimensional cube [-1,1]^m
    x = np.random.uniform(-1, 1, (N, m))
    
    # Evaluate function at these points
    fun_vals = fun(x) 
    # Compute variance
    var = np.var(fun_vals)
    
    return var

def sobol_effect(input, m, fun, N=1000):
    r"""
        Compute the Sobol's total and main effect of the input on the variance of the polynomial.
        Using Monte Carlo simulation.

        We compute both at the same time, to reuse resources.

        The total effect is defined as:
            $$ ST_j = \frac{1}{2N} \sum_{i=1}^N\left(f(x_j^{(i)}, \mathbf{x}_{\{\sim j\}}^{(i)}) - f(z_j^{(i)}, \mathbf{x}_{\{\sim j\}}^{(i)}) \right)^2 $$
        
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
    return sum(leg_coeffs[1:]**2)

def _var_masked(leg_coeffs, mask):
    """
        Compute the variance of the function based on the Legendre coefficients.
        We use a mask to select the coefficients we want to include in the variance.
    """
    mask[0] = False # we don't want to include the mean in the variance
    if not mask.any():
        return np.nan
    masked_coeffs = leg_coeffs[mask]
    var_sum = sum(masked_coeffs**2)
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


def _get_exp_mask_leq_m(mis, m):
    """
        Get the mask for the exponents that are less than or equal to m.
        This is used to compute the effective dimension of the polynomial.
    """
    mask = np.zeros(mis.exponents.shape[0], dtype=bool)
    for i in range(mis.exponents.shape[0]):
        if np.sum(mis.exponents, axis=0) <= m:
            mask[i] = True
    return mask

def effective_dimension_superposition(mis, leg_coeffs, p=0.99):
    r"""
        Compute the effective dimension of the polynomial based on the Legendre coefficients.
        The effective dimension is defined as the number of inputs that contribute to the variance of the polynomial.
        We compute this by summing the squares of the coefficients and checking if they exceed p.

        The formula for $m_s$ is:
        $$
            m_s = \min\{m: \sum_{|u|\leq m} \sigma_{u}^2 \geq p \sigma^2 \}
        $$
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
    
    start_time = time.time()
    
    poly, unisolvent_node_vals = interpolate(m, n, p, MU)
    interpolate_time = time.time() - start_time
    
    start_time = time.time()
    mis, leg_coeffs = convert_to_legendre(poly, unisolvent_node_vals)
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
    mean_fun = mean_mc(m, MU, N=N)
    var_fun = var_mc(m, MU, N=N)

    importances_mc = dict(
        (i, sobol_effect(i, m, MU, N=N))
        for i in range(m)
    )
    monte_carlo_time = time.time() - start_time

    print(f"\nMean difference rel: {abs(mean_fun - mean_leg(leg_coeffs)) / mean_fun:.6f} = {mean_fun - mean_leg(leg_coeffs):.6f} / {mean_fun:.6f}")
    print(f"Var difference rel: {abs(var_fun - var_leg(leg_coeffs)) / var_fun:.6f} = {var_fun - var_leg(leg_coeffs):.6f} / {var_fun:.6f}")

    print("\nSobol's main effect")
    # Create table headers and data
    headers = ["Input", "Legendre", "Monte Carlo", "Abs Diff"]
    table_data = []
    
    # Add the Sobol's main effect to the table
    for i, st in sorted_main_imp_leg:
        mc = importances_mc[i][0]
        table_data.append([f"{i}", f"{st:.6f}", f"{mc:.6f}", f"{abs(st-mc):.6f}"])
    
    # Print the table using tabulate with markdown format
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    print("\nSobol's total effect")
    # Create table headers and data
    headers = ["Input", "Legendre", "Monte Carlo", "Abs Diff"]
    table_data = []
    
    for i, st in sorted_total_imp_leg:
        mc = importances_mc[i][1] # Monte Carlo total effect
        table_data.append([f"{i}", f"{st:.6f}", f"{mc:.6f}", f"{abs(st-mc):.6f}"])

    print(tabulate(table_data, headers=headers, tablefmt="github"))

    print("\nEffective dimension:")
    print(f"   Superposition: {effective_dimension_superposition(mis, leg_coeffs):.6f}")
    order = np.argsort(-np.array([x[1] for x in total_imp_leg]))
    print(f"   Truncation: {effective_dimension_truncation(mis, leg_coeffs, order):.6f}")


    print(f"\nTiming:")
    print(f"Interpolation: {interpolate_time:.4f} seconds")
    print(f"Convert to Legendre: {legendre_time:.4f} seconds")
    print(f"Sobol calculations: {sobol_time:.4f} seconds")
    print(f"Total Legendre: {interpolate_time + legendre_time + sobol_time:.4f} seconds")
    print(f"Monte Carlo calculations: {monte_carlo_time:.4f} seconds")
    print(f"Total: {interpolate_time + legendre_time + sobol_time + monte_carlo_time:.4f} seconds")

if __name__ == "__main__":
    main()
