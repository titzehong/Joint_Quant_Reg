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

@numba.njit
def covariance_function_single_var(in_1: np.ndarray,
                                   in_2: np.ndarray,
                                   kappa: float,
                                   rho: float,
                                   lambd:float) -> float:
    
    """ GP Covariance Function Single Variable Model

    Args:
        in_1 (np.array): First input
        in_2 (np.array): Second input
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    c_mat = np.array([[1,rho],
                      [rho,1]])
    
    # Use x or condition to get this .....
    i_1 = int(in_1[0])
    i_2 = int(in_2[0])
    
    tau_1 = in_1[1]
    tau_2 = in_2[1]
    
    return (kappa**2)*(c_mat[i_1,i_2])*np.exp(-(lambd**2)*(tau_1-tau_2)**2)

def covariance_function_single_var_vector(knot_points: np.ndarray,
                                            input_tau: np.ndarray,
                                            kappa: float,
                                            rho: float,
                                            lambd:float):
    """ GP Covariance Function with

    Args:
        knot_points (np.array): M x 2 matrix with 1st-column being i index and 2nd column tau values
        input_tau (np.array): t x 2 matrix with 1st-column being i index and 2nd column tau values of desired grid
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    tau_diffs = np.subtract.outer(knot_points[:,1], input_tau[:,1])
    rho_select = np.not_equal.outer(knot_points[:,0], input_tau[:,0])

    return (kappa**2)*(1+rho*rho_select - 1*rho_select)*np.exp(-(lambd**2)*(tau_diffs)**2)

def covariance_mat_single_var(gp_input: np.ndarray,
                   kappa:float,
                   rho:float,
                   lambd:float) -> np.ndarray:
    
    """ Helper function to create a covariance matrix given input

    gp_input (np.array): Array of points to calculate cov matrix, n_samples x 2
    kappa (float): kappa value
    rho (float): rho value
    lambd (float): lambd value

    Returns:
        _type_: Covariance matrix
    """    
    
    metric_args = {'kappa': kappa,
                   'rho': rho,
                   'lambd': lambd}
    
    output_mat = pairwise_distances(gp_input, metric=covariance_function_single_var, **metric_args)
            
    return output_mat

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
def get_interval_vector(tau_in: Union[float, np.ndarray],
                 taus: np.ndarray) -> np.ndarray:
    
    #tau_max = len(taus)
    #out_mat = np.empty_like(tau_in,dtype='int')
    trapz_len = taus[1] - taus[0]
    out_mat = np.ceil((tau_in - taus[0]) / trapz_len)
    out_mat = out_mat.astype(np.int32)
    
    return out_mat


def logistic_transform(tau_input: float,
                       tau_grid: np.ndarray,
                       c_vals_i: np.ndarray)->float:
    """ Calculate the logistic transform of xi for a given tau and zeta function

    Args:
        tau_input (float): Input tau value
        tau_grid (np.ndarray): Grid of tau values
        c_vals_i (np.ndarray): Grid of function evaluations of zeta

    Returns:
        float: _description_
    """

    # Calc normalizing constant
    norm_const = calc_grid_trapezoidal(tau_grid,
                                       c_vals_i,
                                       len(tau_grid)-1)
    
    # Get position where tau input falls on grid
    t_l = get_interval(tau_input, tau_grid)
    t_l_1 = t_l-1

    #e_i
    e_t_l = calc_grid_trapezoidal(tau_grid, c_vals_i, t_l) / norm_const
    

    e_t_l_1 = calc_grid_trapezoidal(tau_grid, c_vals_i, t_l_1) / norm_const

    e_tau_hat = (e_t_l*(tau_input - tau_grid[t_l_1]) + \
                e_t_l_1*(tau_grid[t_l]-tau_input) - \
                (tau_input-tau_grid[t_l_1])*(tau_grid[t_l]-tau_input)*(c_vals_i[t_l]-c_vals_i[t_l_1])) / \
                (tau_grid[t_l] - tau_grid[t_l_1])
    
    return e_tau_hat


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
            return e_t_l-diff # Specific edge case when tau input is at left boundary 

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
    
    return e_tau_hat


def base_quantile_function(tau: float,
                      mean: float,
                      sd: float,
                      v: int=1,
                      dist: str='norm') -> float:
    """ Base quantile function

    Args:
        tau (float): input quantile
        mean (float): mean of distribution
        sd (float): sd of distribution
        v (int, optional): DOF if dist='t'
        dist (str, optional): Distribution type.

    Returns:
        float: _description_
    """


    if dist=='norm':
        return norm.ppf(tau, mean, sd)
    
    elif dist == 't':
        return t.ppf(tau, df=v, loc=mean, scale=sd)
    
    else:
        print('Error')


def eta_function_i(tau_input: np.ndarray,
                 w_vals: np.ndarray,
                 tau_grid: np.ndarray,
                 mean: float,
                 sd: float,
                 v: int,
                 sigma: float,
                 dist: str='norm') -> float:
    
    
    # Calculate xi function from GP inputs w
    c_vals = np.exp(w_vals)
    
    # Apply logistic transform
    if len(c_vals.shape) > 1:  # If multiple samples of w_vals inputted (useful for plotting)
        
        # if tau_input is a vector
        if hasattr(tau_input, "__len__"): # if multiple tau values
            xi_vals = np.array([[logistic_transform(t, tau_grid, c_vals[s,:]) for t in tau_input]
                                for s in range(c_vals.shape[0])])
        else: # if single tau value
            xi_vals = np.array([logistic_transform(tau_input, tau_grid, c_vals[s,:]) for s in range(c_vals.shape[0])])
            
    else:  # if single w val (as in MCMC loop)
        if hasattr(tau_input, "__len__"): # if multiple tau values
            xi_vals = np.array([logistic_transform(t, tau_grid, c_vals) for t in tau_input])  
        else:  # if single tau value
            xi_vals = logistic_transform(tau_input, tau_grid, c_vals)
    
    # Apply base quantile function
    eta_out = sigma * base_quantile_function(xi_vals,
                                                 mean,
                                                 sd,
                                                 v=v,
                                                 dist=dist)
    
    return eta_out


def Q_joint_quantile_function(tau_input: float,
                      x_vals: np.ndarray,
                      w_samples_1: np.ndarray,
                      w_samples_2: np.ndarray,
                      sigma_1: float,
                      sigma_2: float,
                      tau_grid: np.ndarray,
                      mu: float,
                      gamma: float,
                      base_quantile_mean: float,
                      base_quantile_sd: float,
                      base_quantile_v: int=1,
                      base_quantile_dist: str='norm') -> np.ndarray:
    """_summary_

    Args:
        tau_input (float): _description_
        x_vals (np.ndarray): _description_
        w_samples_1 (np.ndarray): _description_
        w_samples_2 (np.ndarray): _description_
        sigma_1 (float): _description_
        sigma_2 (float): _description_
        tau_grid (np.ndarray): _description_
        mu (float): _description_
        gamma (float): _description_
        base_quantile_mean (float): _description_
        base_quantile_sd (float): _description_
        base_quantile_v (int, optional): _description_. Defaults to 1.
        base_quantile_dist (str, optional): _description_. Defaults to 'norm'.

    Returns:
        np.ndarray: _description_
    """

    
    eta_out_1 = eta_function_i(tau_input=tau_input,
                               w_vals=w_samples_1,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_1,
                               dist=base_quantile_dist)
    
    eta_out_2 = eta_function_i(tau_input=tau_input,
                               w_vals=w_samples_2,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_2,
                               dist=base_quantile_dist)
    
    output = mu + gamma*x_vals + \
        ((1-x_vals)/2)*eta_out_1 + \
        ((1+x_vals)/2)*eta_out_2
    
    return output



## GP Approximation as in Tokdar 2007
def covariance_function_single_var_vector(knot_points: np.ndarray,
                                            input_tau: np.ndarray,
                                            kappa: float,
                                            rho: float,
                                            lambd:float):
    """ GP Covariance Function with

    Args:
        knot_points (np.array): M x 2 matrix with 1st-column being i index and 2nd column tau values
        input_tau (np.array): t x 2 matrix with 1st-column being i index and 2nd column tau values of desired grid
        kappa (float): kappa value
        rho (float): rho value
        lambd (float): lambd value

    Returns:
        _type_: Covariance
    """    

    tau_diffs = np.subtract.outer(knot_points[:,1], input_tau[:,1])
    rho_select = np.not_equal.outer(knot_points[:,0], input_tau[:,0])

    return (kappa**2)*(1+rho*rho_select - 1*rho_select)*np.exp(-(lambd**2)*(tau_diffs)**2)

def calc_knot_approx(tau_in,
                     knot_points_t,
                     cov_mat_knots,
                     X_surrogates,
                     kappa,
                     rho,
                     lambd):
    
    m = len(knot_points_t)
    
    # Calc covariance matrix 
    knot_sub_ids = np.array(m*[0] + m*[1]).reshape(-1,1)
    knot_points_t = np.concatenate([knot_points_t,
                                knot_points_t]).reshape(-1,1)

    knot_points = np.hstack([knot_sub_ids, knot_points_t])


     # vector
    cov_input_knots = covariance_function_single_var_vector(knot_points,
                                            tau_in,
                                            kappa,
                                            rho,
                                            lambd)
    
    
    A_mat = np.linalg.inv(sp.linalg.sqrtm(cov_mat_knots)) @ cov_input_knots


    f_xA = X_surrogates@A_mat
    
    return f_xA


def calc_knot_approx_v2(tau_in,
                        knot_points_t,
                        cov_mat_knots,
                        w_knot_points,
                        kappa,
                        rho,
                        lambd):
    
    # Uses W evaluated at knot points
    m = len(knot_points_t)
    
    # Calc covariance matrix 
    knot_sub_ids = np.array(m*[0] + m*[1]).reshape(-1,1)
    knot_points_t = np.concatenate([knot_points_t,
                                knot_points_t]).reshape(-1,1)

    knot_points = np.hstack([knot_sub_ids, knot_points_t])


     # vector 
    cov_input_knots = covariance_function_single_var_vector(knot_points,
                                            tau_in,
                                            kappa,
                                            rho,
                                            lambd)
    
    
    f_w_approx = w_knot_points @ np.linalg.inv(cov_mat_knots) @ cov_input_knots


    return f_w_approx


@numba.njit
def eta_function_i_vector(tau_input: np.ndarray,
                 w_vals: np.ndarray,
                 tau_grid: np.ndarray,
                 mean: float,
                 sd: float,
                 v: int,
                 sigma: float,
                 dist: str='norm'):
    
    
    c_vals = np.exp(w_vals)
    
    # Apply logistic transform

    xi_vals = logistic_transform_vector(tau_input, tau_grid, c_vals)

    # Apply 

    if dist == 'norm':
        eta_out = sigma * numba_norm.ppf(xi_vals,
                                     mean,
                                     sd)
    
    
    return eta_out

@numba.njit
def Q_joint_quantile_function_vector(tau_input: np.ndarray,
                      x_vals: np.ndarray,
                      w_samples_1: np.ndarray,
                      w_samples_2: np.ndarray,
                      sigma_1: float,
                      sigma_2: float,
                      tau_grid: np.ndarray,
                      mu: float,
                      gamma: float,
                      base_quantile_mean: float,
                      base_quantile_sd: float,
                      base_quantile_v: int=1,
                      base_quantile_dist: str='norm'):
    
    
    eta_out_1 = eta_function_i_vector(tau_input=tau_input,
                               w_vals=w_samples_1,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_1,
                               dist=base_quantile_dist)
    
    eta_out_2 = eta_function_i_vector(tau_input=tau_input,
                               w_vals=w_samples_2,
                               tau_grid=tau_grid,
                               mean=base_quantile_mean,
                               sd=base_quantile_sd,
                               v=base_quantile_v,
                               sigma=sigma_2,
                               dist=base_quantile_dist)
    
    output = mu + gamma*x_vals + \
        ((1-x_vals)/2)*eta_out_1 + \
        ((1+x_vals)/2)*eta_out_2
    
    return output

@numba.njit
def grid_search_deriv_approx_vector(y_i:float,
                             x_i: np.ndarray,
                             w_samples_1: np.ndarray,
                             w_samples_2: np.ndarray,
                             sigma_1: float,
                             sigma_2: float,
                             tau_grid: np.ndarray,
                             tau_grid_expanded: np.ndarray,
                             mu: float,
                             gamma: float,
                             base_quantile_mean: float,
                             base_quantile_sd: float,
                             base_quantile_v: int=1,
                             base_quantile_dist: str='norm'):

    
    
    Q_y_i_vals = Q_joint_quantile_function_vector(tau_input=tau_grid,
                                              x_vals=x_i,
                                              w_samples_1=w_samples_1,
                                              w_samples_2=w_samples_2,
                                              sigma_1=sigma_1,
                                              sigma_2=sigma_2,
                                              tau_grid=tau_grid_expanded,
                                              mu=mu,
                                              gamma=gamma,
                                              base_quantile_mean=base_quantile_mean,
                                              base_quantile_sd=base_quantile_sd,
                                              base_quantile_v=base_quantile_v,
                                              base_quantile_dist=base_quantile_dist)
    
    
    t_l = 0
    while True:
        if Q_y_i_vals[t_l] > y_i:
            
            break
        t_l += 1
        if t_l == len(tau_grid):
            break

    if t_l >= len(tau_grid)-1:
        tau_edge = 0.99999
        """
        Q_y_edge = Q_joint_quantile_function_vector(tau_input=np.array([tau_edge]),
                                            x_vals=x_i,
                                            w_samples_1=w_samples_1,
                                            w_samples_2=w_samples_2,
                                            sigma_1=sigma_1,
                                            sigma_2=sigma_2,
                                            tau_grid=tau_grid_expanded,
                                            mu=mu,
                                            gamma=gamma,
                                            base_quantile_mean=base_quantile_mean,
                                            base_quantile_sd=base_quantile_sd,
                                            base_quantile_v=base_quantile_v,
                                            base_quantile_dist=base_quantile_dist) 

        top_diff = Q_y_edge[0] - Q_y_i_vals[-2]
        if top_diff == 0:
            top_diff = 1e-20
        """
        #top_diff = 0.009990000000000054
        top_diff = 0.1
        deriv_Q_y = (top_diff)/(tau_edge - tau_grid[t_l-1])
    
    
    elif t_l == 0:
        tau_edge = 0.005
        """
        Q_y_edge = Q_joint_quantile_function_vector(tau_input=np.array([tau_edge]),
                                    x_vals=x_i,
                                    w_samples_1=w_samples_1,
                                    w_samples_2=w_samples_2,
                                    sigma_1=sigma_1,
                                    sigma_2=sigma_2,
                                    tau_grid=tau_grid_expanded,
                                    mu=mu,
                                    gamma=gamma,
                                    base_quantile_mean=base_quantile_mean,
                                    base_quantile_sd=base_quantile_sd,
                                    base_quantile_v=base_quantile_v,
                                    base_quantile_dist=base_quantile_dist) 
        
        """
        bot_diff = 0.05
        deriv_Q_y = (bot_diff)/(tau_grid[t_l] - tau_edge)
        #print(tau_grid[t_l])
    
    else:
        deriv_Q_y = (Q_y_i_vals[t_l] - Q_y_i_vals[t_l-1])/(tau_grid[t_l] - tau_grid[t_l-1])
        
        #print(Q_y_i_vals[t_l])
        #print(Q_y_i_vals[t_l-1])
        #print(t_l)
        #print(t_l-1)
        
    #print(tau_grid[t_l])
    return deriv_Q_y

@numba.njit(parallel=True)
def eval_ll(y_vals_true,
           x_vals,
            w_samples_1,
          w_samples_2,
          sigma_1,
          sigma_2,
          tau_grid,
          tau_grid_expanded,
          mu,
          gamma,
          base_quantile_mean=0.0,
          base_quantile_sd=1.0,
          base_quantile_v=1,
          base_quantile_dist='norm'):

    log_lik = 0

    for i in prange(1,len(y_vals_true)):

        y_i = y_vals_true[i]
        x_i = x_vals[i]

        ll_i = grid_search_deriv_approx_vector(y_i=y_i,
                              x_i=x_i,
                              w_samples_1=w_samples_1,
                              w_samples_2=w_samples_2,
                              sigma_1=sigma_1,
                              sigma_2=sigma_2,
                              tau_grid=tau_grid,
                              tau_grid_expanded=tau_grid_expanded,
                              mu=mu,
                              gamma=gamma,
                              base_quantile_mean=base_quantile_mean,
                              base_quantile_sd=base_quantile_sd,
                              base_quantile_v=base_quantile_v,
                              base_quantile_dist=base_quantile_dist)

        log_lik += np.log(ll_i)

    log_lik = -1 * log_lik
    
    return log_lik


def metropolis_step_kappa(kappa_current,
                          mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           rho_current,
                           lambda_current,
                           knot_points,
                           w_approx_current,
                           w_X_current,
                           tau_input,
                           y_vals,
                           x_vals):

    # Sample new value for kappa
    kappa_inv_sq_prop = gamma.rvs(a=3, scale=3)
    kappa_prop = kappa_inv_sq_prop**(-1/2)

    cov_mat_knots_prop = covariance_mat_single_var(knot_points,
                             kappa=kappa_prop,
                             rho=rho_current,
                             lambd=lambda_current)

    w_approx_prop = calc_knot_approx(tau_input,
                                    np.arange(0.1,1,0.1),
                                    cov_mat_knots_prop,
                                    w_X_current,
                                    kappa_prop,
                                    rho_current,
                                    lambda_current)
    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=w_approx_prop[0:100],
                        w_samples_2=w_approx_prop[100:200],
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_input,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=w_approx_current[0:100],
                        w_samples_2=w_approx_current[100:200],
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_input,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = gamma.logpdf(kappa_prop**(-1/2),  a=3, scale=3)
    prior_lp_curr = gamma.logpdf(kappa_current**(-1/2),  a=3, scale=3)


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return kappa_prop, 1
    else: 
        return kappa_current, 0


def metropolis_step_kappa(kappa_current,
                          mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           rho_current,
                           lambda_current,
                           knot_points,
                           w_approx_current,
                           w_X_current,
                           tau_input,
                           tau_grid,
                           y_vals,
                           x_vals):
    
    
    # Sample new value for kappa
    kappa_inv_sq_prop = gamma.rvs(a=3, scale=3)
    kappa_prop = kappa_inv_sq_prop**(-1/2)

    cov_mat_knots_prop = covariance_mat_single_var(knot_points,
                             kappa=kappa_prop,
                             rho=rho_current,
                             lambd=lambda_current)

    w_approx_prop = calc_knot_approx(tau_input,
                                    np.arange(0.1,1,0.1),
                                    cov_mat_knots_prop,
                                    w_X_current,
                                    kappa_prop,
                                    rho_current,
                                    lambda_current)
    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_prop[0:100]),
                        w_samples_2=np.real(w_approx_prop[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = gamma.logpdf(kappa_prop**(2),  a=3, scale=3)
    prior_lp_curr = gamma.logpdf(kappa_current**(2),  a=3, scale=3)


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return kappa_prop, 1
    else: 
        return kappa_current, 0




def metropolis_step_lambda(kappa_current,
                          mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           rho_current,
                           lambda_current,
                           knot_points,
                           w_approx_current,
                           w_X_current,
                           tau_input,
                           tau_grid,
                           y_vals,
                           x_vals):

    # Sample new value for kappa
    lambda_sq_prop = gamma.rvs(a=5, scale=10)
    lambda_prop = np.sqrt(lambda_sq_prop)


    cov_mat_knots_prop = covariance_mat_single_var(knot_points,
                             kappa=kappa_current,
                             rho=rho_current,
                             lambd=lambda_prop)

    w_approx_prop = calc_knot_approx(tau_input,
                                    np.arange(0.1,1,0.1),
                                    cov_mat_knots_prop,
                                    w_X_current,
                                    kappa_current,
                                    rho_current,
                                    lambda_prop)
    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_prop[0:100]),
                        w_samples_2=np.real(w_approx_prop[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = gamma.logpdf(lambda_prop**(2),  a=5, scale=10)
    prior_lp_curr = gamma.logpdf(lambda_current**(2),  a=5, scale=10)


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return lambda_prop, 1
    else: 
        return lambda_current, 0


def metropolis_step_rho(kappa_current,
                          mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           rho_current,
                           lambda_current,
                           knot_points,
                           w_approx_current,
                           w_X_current,
                           tau_input,
                           tau_grid,
                           y_vals,
                           x_vals):

    # Sample new value for kappa
    rho_prop = uniform.rvs(0,1)


    cov_mat_knots_prop = covariance_mat_single_var(knot_points,
                             kappa=kappa_current,
                             rho=rho_prop,
                             lambd=lambda_current)

    w_approx_prop = calc_knot_approx(tau_input,
                                    np.arange(0.1,1,0.1),
                                    cov_mat_knots_prop,
                                    w_X_current,
                                    kappa_current,
                                    rho_prop,
                                    lambda_current)
    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_prop[0:100]),
                        w_samples_2=np.real(w_approx_prop[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = 1
    prior_lp_curr = 1


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return rho_prop, 1
    else: 
        return rho_current, 0



def metropolis_step_sigmas(mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           w_approx_current,
                           tau_grid,
                           y_vals,
                           x_vals):

    # Sample new value for kappa
    sigma_1_prop = gamma.rvs(a=2, scale=1/2)
    sigma_2_prop = gamma.rvs(a=2, scale=1/2)

    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_prop,
                        sigma_2=sigma_2_prop,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = gamma.logpdf(sigma_1_prop, a=2, scale=1/2) + gamma.logpdf(sigma_2_prop, a=2, scale=1/2) 
    prior_lp_curr = gamma.logpdf(sigma_1_current, a=2, scale=1/2) + gamma.logpdf(sigma_2_current, a=2, scale=1/2) 


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return sigma_1_prop, sigma_2_prop,  1
    else: 
        return sigma_1_current, sigma_2_current, 0
    

def metropolis_step_mu(mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           w_approx_current,
                           tau_grid,
                           y_vals,
                           x_vals):

    # Sample new value for kappa
    mu_prop = norm.rvs(0,1)

    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_prop,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = norm.logpdf(mu_prop, 0,1)
    prior_lp_curr = norm.logpdf(mu_current, 0,1)


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return mu_prop,  1
    else: 
        return mu_current, 0
    

def metropolis_step_gamma(mu_current,
                          gamma_current,
                           sigma_1_current,
                           sigma_2_current,
                           w_approx_current,
                           tau_grid,
                           y_vals,
                           x_vals):

    # Sample new value for kappa from proposal
    gamma_prop = norm.rvs(0,1)

    
    # Calc likelihood
    ll_prop = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_prop,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')
    
    ll_curr = eval_ll(y_vals,
                        x_vals,
                        w_samples_1=np.real(w_approx_current[0:100]),
                        w_samples_2=np.real(w_approx_current[100:200]),
                        sigma_1=sigma_1_current,
                        sigma_2=sigma_2_current,
                        tau_grid=tau_grid,
                        mu=mu_current,
                        gamma=gamma_current,
                        base_quantile_mean=0.0,
                        base_quantile_sd=1.0,
                        base_quantile_v=1,
                        base_quantile_dist='norm')

    # Take metropolis step
    prior_lp_prop = norm.logpdf(gamma_prop, 0,1)
    prior_lp_curr = norm.logpdf(gamma_current, 0,1)


    a = np.exp(min(0, prior_lp_prop+ll_prop -\
                prior_lp_curr+ll_curr))
    
    if np.random.uniform(0,1) < a: # accepted
        return gamma_prop,  1
    else: 
        return gamma_current, 0
    

def generate_beta_samples(tau_input: float,
                          tau_grid: np.ndarray,
                          w_approx_store: List[np.ndarray],
                          mu_store: List[float],
                          gamma_store: List[float],
                          sigma_1_store: List[float],
                          sigma_2_store: List[float]):

    beta_0_store = []
    beta_1_store = []

    for i in range(0,len(w_approx_store)):

        w_samp = w_approx_store[i]
        w1_samp = w_samp[0:100]
        w2_samp = w_samp[100:200]

        mu_samp = mu_store[i]
        gamma_samp = gamma_store[i]
        sigma_1_samp = sigma_1_store[i]
        sigma_2_samp = sigma_2_store[i]


        eta_1_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w1_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=1,
                                             sigma=sigma_1_samp,
                                             dist='norm')[0]


        eta_2_samp = eta_function_i_vector(tau_input=np.array([tau_input]),
                                             w_vals=w2_samp,
                                             tau_grid=tau_grid,
                                             mean=0.0,
                                             sd=1.0,
                                             v=1,
                                             sigma=sigma_2_samp,
                                             dist='norm')[0]


        beta_0_samp = mu_samp + (eta_1_samp + eta_2_samp)/2
        beta_1_samp = gamma_samp + (eta_2_samp - eta_1_samp)/2

        beta_0_store.append(beta_0_samp)
        beta_1_store.append(beta_1_samp)
        
        return beta_0_store, beta_1_store
    

def logpdf_mvn(x, mean, cov):
    """
    Compute the loglikelihood of a multivariate normal distribution (MND).
    
    Parameters:
    x (numpy array): A 1-D numpy array of data points.
    mean (numpy array): A 1-D numpy array representing the mean vector of the MND.
    cov (numpy array): A 2-D numpy array representing the covariance matrix of the MND.
    
    Returns:
    float: The loglikelihood of the MND.
    """
    n = x.shape[0]
    diff = x - mean
    return -0.5 * (n * np.log(2 * np.pi) + np.log(np.linalg.det(cov)) + diff.T @ np.linalg.inv(cov) @ diff)
