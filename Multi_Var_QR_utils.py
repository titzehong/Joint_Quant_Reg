import numpy as np
import matplotlib.pyplot as plt
import numba 
from sklearn.metrics import pairwise_distances
from scipy.stats import norm, t, gamma, uniform
from typing import Union
import scipy as sp
from scipy.stats import norm
from numba_stats import norm as numba_norm
from numba_stats import t as numba_t
from numba import prange
from typing import List
from   scipy.special import gammaln


def covariance_mat_single_var(gp_input: np.ndarray,
                   kappa:float,
                   lambd:float,
                   with_kappa:bool=True) -> np.ndarray:
    
    """ Helper function to create a covariance matrix given input

    gp_input (np.array): Array of points to calculate cov matrix, n_samples x 2
    kappa (float): kappa value
    rho (float): rho value
    lambd (float): lambd value

    Returns:
        _type_: Covariance matrix
    """    
    
    metric_args = {'kappa': kappa,
                   'lambd': lambd,
                    'with_kappa': with_kappa}
    
    output_mat = pairwise_distances(gp_input, metric=covariance_function_single_var, **metric_args)
            
    return output_mat


@numba.njit
def covariance_function_single_var(tau_1: np.ndarray,
                                   tau_2: np.ndarray,
                                   kappa: float,
                                   lambd:float,
                                   with_kappa:float = True) -> float:
    
    """ GP Covariance Function Single Variable Model

    Args:
        tau_1 (np.array): First input
        tau_2 (np.array): Second input
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    
    
    if with_kappa:
        return (kappa**2)*np.exp(-(lambd**2)*(tau_1-tau_2)**2)
    else:
        return np.exp(-(lambd**2)*(tau_1-tau_2)**2)
    

@numba.njit
def logsumexp(x):
    # Utility to do renormalization properly
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

@numba.njit
def renorm_dist(log_p_vec):
    return np.exp(log_p_vec - logsumexp(log_p_vec))


def logpdf_t(x, mean, shape, df):
    # https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    dim = mean.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E


def precompute_approx(tau_grid: np.ndarray,
                      knot_points_grid: np.ndarray,
                      lambda_grid: np.ndarray):
    

    # Pre-compute things
    cov_mat_knot_store = []
    A_g_matrices = []
    G = len(lambda_grid)

    # Pre compute stuff
    for g in range(G):
        lambda_g = lambda_grid[g]
        cov_mat_knots_g = covariance_mat_single_var(knot_points_grid.reshape(-1,1), 0,
                                                    lambda_g,
                                                    with_kappa=False)
        
        cov_mat_knot_store.append(cov_mat_knots_g)
        
        # compute C_0
        C_0 = np.zeros([len(tau_grid), len(knot_points_grid)])
        for l in range(len(tau_grid)):
            for m in range(len(knot_points_grid)):
                tau_l = tau_grid[l]
                tau_m = knot_points_grid[m]
                C_0[l,m] = covariance_function_single_var(tau_l,
                                                    tau_m,
                                                    kappa=0, lambd = lambda_g, with_kappa=False)

        cov_mat_knots_inv = np.linalg.inv(cov_mat_knots_g)
        A_g = C_0 @ cov_mat_knots_inv
        A_g_matrices.append(A_g)

    return cov_mat_knot_store, A_g_matrices 

def calc_knot_approx_marginalized(w_j_star: np.ndarray,
                                 a_kappa: float,
                                 b_kappa: float,
                                 tau_grid: np.ndarray,
                                 A_g_matrices: List[np.ndarray],
                                 cov_mat_knot_store: List[np.ndarray],
                                 lambda_grid_log_prob: List[float]):
    
    G = len(A_g_matrices)
    M = len(w_j_star)

    norm_pdf_lambda = renorm_dist(lambda_grid_log_prob)

    t_log_pdfs = np.zeros(G)
    for g in range(G):
        t_log_pdf = logpdf_t(w_j_star,
                            np.zeros(M),
                            (b_kappa/a_kappa)*cov_mat_knot_store[g],
                            2*a_kappa)

        t_log_pdfs[g] = t_log_pdf
        
    output_vec = np.zeros(len(tau_grid))
    matrix_term = np.zeros([A_g_matrices[0].shape[0], A_g_matrices[0].shape[1]])
    marginal_log_prob = 0
    for g in range(G):
        
        matrix_term += norm_pdf_lambda[g] * np.exp(t_log_pdfs[g]) * (A_g_matrices[g])

        marginal_log_prob += np.log(norm_pdf_lambda[g]) + t_log_pdfs[g]

    output_vec = matrix_term @ w_j_star 
    return output_vec, marginal_log_prob


@numba.njit
def get_interval(tau_in: Union[float, np.ndarray] ,
                 tau_grid: np.ndarray) -> int:
    """ Finds the upper index of the interval in tau_grid where tau_in is located

    Args:
        tau_in (float): input tau value
        tau_grid (np.ndarray): Grid of tau values

    Returns:
        int: t_l if t \in [tau_grid[t_l-1], tau_grid[t_l]]
    """
        
    trapz_len = tau_grid[1] - tau_grid[0]

    return np.ceil((tau_in - tau_grid[0]) / trapz_len)

@numba.njit
def calculate_contiguous_row_sums_numba(in_matrix,
                                        start_column_indices,
                                        end_column_indices):
    row_sums = np.zeros(len(in_matrix))
    
    for i in prange(len(in_matrix)):
        start = start_column_indices[i]
        end = end_column_indices[i]
        row_sums[i] = sum(in_matrix[i, start:end])
    
    return row_sums



@numba.njit
def calc_grid_trapezoidal_vector(tau_grid,
                                  c_vals,
                                  c_samp_repeat,
                                 last_ids):
    
    #print('as')

    trapz_len = tau_grid[1] - tau_grid[0]

    #c_samp_repeat = np.repeat(c_vals[:,np.newaxis],len(tau_grid), axis=1).T
    
    
    mid_vals = calculate_contiguous_row_sums_numba(c_samp_repeat,
                                                   np.ones(len(last_ids), dtype='int'),
                                                   last_ids)

    trapz_sum = (trapz_len/2)*(c_vals[0] + 2*mid_vals + c_vals[last_ids])
    return trapz_sum
    
@numba.njit
def calc_grid_trapezoidal(tau_grid: np.ndarray, c_vals:np.ndarray, last_id: int) -> float:    
    """ Calculates trapezoidal approximation for a given grid and values

    Args:
        taus (np.ndarray): Grid of tau values
        c_vals (np.ndarray): Function evaluation at each point in taus
        last_id (int): the index of point in tau grid to numerically integrate until

    Returns:
        float: _description_
    """
    trapz_len = tau_grid[1] - tau_grid[0]
    # Sep7 to fix!! Bug if last id = 0 this double counts

    trapz_sum = (trapz_len/2)*(c_vals[0] + 2*c_vals[1:last_id].sum() + c_vals[last_id])
    
    return trapz_sum



@numba.njit
def logistic_transform_vector(tau_input: Union[float, np.ndarray],
                       tau_grid_expanded: np.ndarray,
                       c_vals_i: np.ndarray):
    
    # Calc grid distance
    trapz_len = tau_grid_expanded[1] - tau_grid_expanded[0]
    
    # Calc normalizing constant
    norm_const = calc_grid_trapezoidal(tau_grid_expanded,
                                       c_vals_i,
                                       len(tau_grid_expanded)-1)
    
    # Get position where tau input falls on grid
    t_ls = get_interval_vector(tau_input, tau_grid_expanded).astype('int')
    t_ls_1 = t_ls-1
    
    """
    if hasattr(tau_input, "__len__"):
        t_ls = t_ls.astype('int')
    else:
        t_ls = int(t_ls)
    
    t_ls_1 = t_ls-1
    """

    #e_i
    #c_samp_repeat = np.repeat(c_vals_i[:,np.newaxis],len(tau_input), axis=1).T
    c_samp_repeat = np.repeat(c_vals_i,len(tau_input)).reshape((len(c_vals_i), 
                                                    len(tau_input))).T
    
    e_t_l = calc_grid_trapezoidal_vector(tau_grid_expanded, c_vals_i, c_samp_repeat, t_ls) / norm_const

    if len(t_ls) == 1:
        if t_ls[0] == 0:
            diff = e_t_l / 2
            return e_t_l-diff, c_vals_i/norm_const  # Specific edge case when tau input is at left boundary 

    e_t_l_1 = np.zeros(len(e_t_l)) 
    e_t_l_1[1:-1] =  e_t_l[0:len(e_t_l)-2]
    #e_t_l_1 = np.concatenate([np.array([0.0]),e_t_l[0:len(e_t_l)-1]])

    #e_t_l_1 = calc_grid_trapezoidal_vector(tau_grid_expanded, c_vals_i, c_samp_repeat, t_ls_1) / norm_const

    """
    e_tau_hat = (e_t_l*(tau_input - tau_grid[t_ls_1]) + \
                e_t_l_1*(tau_grid[t_ls]-tau_input) - \
                (tau_input-tau_grid[t_ls_1])*(tau_grid[t_ls]-tau_input)*(c_vals_i[t_ls]-c_vals_i[t_ls_1])) / \
                (tau_grid[t_ls] - tau_grid[t_ls_1])
    """
    e_tau_hat = (e_t_l*(tau_input - tau_grid_expanded[t_ls_1]) + \
                e_t_l_1*(tau_grid_expanded[t_ls]-tau_input) - \
                (tau_input-tau_grid_expanded[t_ls_1])*(tau_grid_expanded[t_ls]-tau_input)*(e_t_l-e_t_l_1)) / \
                (tau_grid_expanded[t_ls] - tau_grid_expanded[t_ls_1])
    

    
    return e_tau_hat, c_vals_i/norm_const


@numba.njit
def get_interval_vector(tau_in: Union[float, np.ndarray],
                 taus: np.ndarray) -> np.ndarray:
    
    #tau_max = len(taus)
    #out_mat = np.empty_like(tau_in,dtype='int')
    trapz_len = taus[1] - taus[0]
    out_mat = np.ceil((tau_in - taus[0]) / trapz_len)
    out_mat = out_mat.astype(np.int32)
    
    return out_mat