import numpy as np
import matplotlib.pyplot as plt
import numba 
from numba import prange
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, gamma, multivariate_t
from scipy.stats import t as student_t
import time


from Single_Var_QR_utils import *



class SingleQRSampler:

    def __init__(self, 
                y_vals_true,
                x_vals,          
                C_1 = 0.5,
                lambda_step_size_1 = 3,
                alpha_step_size_1 = 0.45,
                a_target_1 = 0.4,
                C_2 = 0.5,
                lambda_step_size_2 = 3,
                alpha_step_size_2 = 0.45,
                a_target_2 = 0.4,
                tau_grid_expanded = np.arange(-0.01,1.02,0.01),
                tau_grid = np.arange(0.01,1.0,0.01),     
                knot_points_grid = np.arange(0.1,1,0.1),
                am_lamb_block1_init = 0.1,
                am_lamb_block2_init = 0.5,
                alpha_kappa = 5,
                beta_kappa = 1/3,
                eps_1 = 1e-5,
                eps_2 = 1e-5,
                base_quantile_mean=0.0,
                base_quantile_sd=1.0,
                base_quantile_v=1,
                base_quantile_dist='norm'):
        
        # AM Sampler hyper params Block 1
        self.C_1 = C_1
        self.lambda_step_size_1 =lambda_step_size_1
        self.alpha_step_size_1 = alpha_step_size_1
        self.a_target_1 = a_target_1
        self.am_lamb_block1_init =am_lamb_block1_init
        self.eps_1 = eps_1
        self.alpha_kappa = alpha_kappa
        self.beta_kappa = beta_kappa

        # AM Sampler hyper params Block 2
        self.C_2 =C_2
        self.lambda_step_size_2 =lambda_step_size_2
        self.alpha_step_size_2 = alpha_step_size_2
        self.a_target_2 =a_target_2
        self.am_lamb_block2_init = am_lamb_block2_init
        self.eps_2 = eps_2

        # Tau grid 
        self.tau_grid_expanded =tau_grid_expanded
        self.tau_grid = tau_grid
        self.knot_points_grid =knot_points_grid

        # Quantile function parameters
        self.base_quantile_mean = base_quantile_mean
        self.base_quantile_sd = base_quantile_sd
        self.base_quantile_v = base_quantile_v
        self.base_quantile_dist = base_quantile_dist

        # Storage variables
        self.w_approx_store = []
        self.kappa_store = []
        self.lambda_store = []
        self.rho_store = []

        self.sigma_1_store = []
        self.sigma_2_store = []
        self.mu_store = []
        self.gamma_store = []


        # Data
        self.y_vals_true = y_vals_true
        self.x_vals = x_vals

        self.block_1_accept_cnts = 0
        self.block_2_accept_cnts = 0

        # Store stuff for logging
        self.ll_check_blk1 = []
        self.ll_check_blk2 = []
        self.prop_check1 = []
        self.prop_check2 = []
        self.a_check_1 = []
        self.cov_store_2 = []
        self.mu_block_store_2 =[]
        self.am_params_store_2 = []
        self.a_check_2 = []
        self.blk2_check = []
        self.log_score = []

    def sample(self, n_steps=20000, n_burn_in=5000):

        # Start time
        s = time.time()

        self.n_steps = n_steps

        # Initialize Step sizes
        step_sizes_1 = self.C_1/(np.arange(1,n_steps+10)**self.alpha_step_size_1)
        step_sizes_2 = self.C_2/(np.arange(1,n_steps+10)**self.alpha_step_size_2)


        # Calc covariance matrix 
        knot_points_t = self.knot_points_grid
        m = len(knot_points_t)  # no knots
        knot_sub_ids = np.array(m*[0] + m*[1]).reshape(-1,1)
        knot_points_t = np.concatenate([knot_points_t,
                                    knot_points_t]).reshape(-1,1)

        knot_points = np.hstack([knot_sub_ids, knot_points_t])

        tau_input_test_0 = np.vstack([np.zeros(len(self.tau_grid)), self.tau_grid]).T
        tau_input_test_1 = np.vstack([np.ones(len(self.tau_grid)), self.tau_grid]).T
        tau_input = np.vstack([tau_input_test_0, tau_input_test_1])

        tau_input_test_0_expanded = np.vstack([np.zeros(len(self.tau_grid_expanded)),
                                            self.tau_grid_expanded]).T
        tau_input_test_1_expanded= np.vstack([np.ones(len(self.tau_grid_expanded)),
                                            self.tau_grid_expanded]).T

        tau_input_expanded = np.vstack([tau_input_test_0_expanded,
                                        tau_input_test_1_expanded])

        #### Initialize Model Parameters
        ## GP Related hyperparameters
        kappa_current = np.nan
        rho_current = 0.1
        lambd_current = 4
        alpha_kappa = 3
        beta_kappa = 1/3


        ## Regression related parametrs
        mu_current = 0
        gamma_current = 0
        sigma_1_current = 1
        sigma_2_current = 1


        #### W samples
        # calc covariance matrix
        cov_mat_knots_current = covariance_mat_single_var(knot_points,
                                    kappa=kappa_current,
                                    rho=rho_current,
                                    lambd=lambd_current,
                                    with_kappa=False)

        
        w_knot_points_current = multivariate_t.rvs(loc=np.zeros(m*2),
                              shape=cov_mat_knots_current*(beta_kappa/alpha_kappa),
                              df=2*alpha_kappa)


        # Generate sample of GP approx
        w_approx_current = calc_knot_approx_v2(tau_input_expanded,
                                        self.knot_points_grid,
                                        cov_mat_knots_current,
                                        w_knot_points_current,
                                        kappa_current,
                                        rho_current,
                                        lambd_current,
                                        with_kappa=False)

        ### initialise adaptive metropolis

        # Block 1
        am_lamb_block1 = self.am_lamb_block1_init
        log_am_lamb_block1 = np.log(am_lamb_block1)
        am_cov_block1 = 10*block_diag(cov_mat_knots_current, np.eye(1))#np.eye(len(w_knot_points_current)+2)

        mu_block1 = np.concatenate([w_knot_points_current,
                                    np.array([lambd_current])])
                                    # np.array([rho_current])])
            
        # Block 2
        am_lamb_block2 = self.am_lamb_block2_init
        log_am_lamb_block2 = np.log(am_lamb_block2)
        am_cov_block2 = 10**np.diag([np.sqrt(5),np.sqrt(5),1,1])
        mu_block2 = np.concatenate([np.array([mu_current]),
                                    np.array([gamma_current]),
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)])])

        # Set up initial state
    
        update_block_1 = np.concatenate([w_knot_points_current,
                                np.array([lambd_current])])

        update_block_2 = np.concatenate([np.array([mu_current]),
                                np.array([gamma_current]),
                                np.array([np.log(sigma_1_current),
                                            np.log(sigma_2_current)])])


        for mc_i in range(self.n_steps): 
                        
            #### Generate Sample for W, kappa, tau, lamdba
            
            am_lamb_block1 = np.exp(log_am_lamb_block1)
            proposal_vec  = np.random.multivariate_normal(update_block_1,  
                                                        am_lamb_block1*am_cov_block1+np.eye(len(mu_block1))*self.eps_1)
            
            w_knot_prop = proposal_vec[0:len(w_knot_points_current)]
            
            lambd_prop = proposal_vec[len(w_knot_points_current)]
            #log_rho_prop = proposal_vec[len(w_knot_points_current)+2]
            
            self.prop_check1.append(proposal_vec)
            
            rho_prop = 0.1 #1/(1+e**log_rho_prop)
            
            # Get new covariance matrix
            cov_mat_knot_prop = covariance_mat_single_var(knot_points,
                                                        kappa=np.nan,
                                                        rho=rho_prop,
                                                        lambd=lambd_prop,
                                                        with_kappa=False)
            
            # Update w_sample
            w_approx_prop = calc_knot_approx_v2(tau_in=tau_input_expanded,
                                                knot_points_t=self.knot_points_grid,
                                                cov_mat_knots=cov_mat_knot_prop,
                                                w_knot_points=w_knot_prop,
                                                kappa=np.nan,
                                                rho=rho_prop,
                                                lambd=lambd_prop,
                                                with_kappa=False)

            # Calc likelihood
            ll_prop = eval_ll(self.y_vals_true,
                            self.x_vals,
                            w_samples_1=w_approx_prop[0:103],
                            w_samples_2=w_approx_prop[103:],
                            sigma_1=sigma_1_current,
                            sigma_2=sigma_2_current,
                            tau_grid=self.tau_grid,
                            tau_grid_expanded=self.tau_grid_expanded,
                            mu=mu_current,
                            gamma=gamma_current,
                            base_quantile_mean=self.base_quantile_mean,
                            base_quantile_sd=self.base_quantile_sd,
                            base_quantile_v=self.base_quantile_v,
                            base_quantile_dist=self.base_quantile_dist)

            ll_curr = eval_ll(self.y_vals_true,
                            self.x_vals,
                            w_samples_1=w_approx_current[0:103],
                            w_samples_2=w_approx_current[103:],
                            sigma_1=sigma_1_current,
                            sigma_2=sigma_2_current,
                            tau_grid=self.tau_grid,
                            tau_grid_expanded=self.tau_grid_expanded,
                            mu=mu_current,
                            gamma=gamma_current,
                            base_quantile_mean=self.base_quantile_mean,
                            base_quantile_sd=self.base_quantile_sd,
                            base_quantile_v=self.base_quantile_v,
                            base_quantile_dist=self.base_quantile_dist)    

            self.ll_check_blk1.append((ll_prop, ll_curr))
            self.log_score.append(ll_curr)
            # Take metropolis step
            
            # Prior Prob
            log_prior_prop = logpdf_t(w_knot_prop,
                                    np.zeros(m*2),
                                    (beta_kappa/alpha_kappa)*cov_mat_knot_prop,
                                    2*alpha_kappa) +\
                            gamma.logpdf(lambd_prop**2,  a=5, scale=10) 
            
            
            log_prior_current = logpdf_t(w_knot_prop,
                                    np.zeros(m*2),
                                    (beta_kappa/alpha_kappa)*cov_mat_knots_current,
                                    2*alpha_kappa) +\
                        gamma.logpdf(lambd_current**2,  a=5, scale=10) 
            
            # Proposal Props
            log_proposal_prop = 0
            
            current_vec = np.concatenate([w_knot_points_current,
                                    np.array([kappa_current]),
                                    np.array([lambd_current])])
            

            log_proposal_current = 0
            
            
            trans_weight_1 = (ll_prop + log_prior_prop + log_proposal_current) - \
                            (ll_curr + log_prior_current + log_proposal_prop) 
            
            if np.isnan(trans_weight_1):
                print('Transition Error Block 1')
                trans_weight_1 = -1e99
            a_1 = np.exp(min(0,  trans_weight_1))
            
            self.a_check_1.append(a_1)
            #print(a)

            if np.random.uniform(0,1) < a_1:
                w_knot_points_current = w_knot_prop
                lambd_current = lambd_prop
                rho_current = rho_prop
                
                self.block_1_accept_cnts += 1
            else: 
                w_knot_points_current = w_knot_points_current
                lambd_current = lambd_current
                rho_current = rho_current
            

            # Update w     
            cov_mat_knots_current = covariance_mat_single_var(knot_points,
                                    kappa=np.nan,
                                    rho=rho_current,
                                    lambd=lambd_current,
                                    with_kappa=False)
            
            w_approx_current = calc_knot_approx_v2(tau_in=tau_input_expanded,
                                        knot_points_t=self.knot_points_grid,
                                        cov_mat_knots=cov_mat_knots_current,
                                        w_knot_points=w_knot_points_current,
                                        kappa=np.nan,
                                        rho=rho_current,
                                        lambd=lambd_current,
                                        with_kappa=False)
            
            # Update AM sampling parameters
            update_block_1 = np.concatenate([w_knot_points_current,
                                    np.array([lambd_current])])
                                    # np.array([np.log(rho_current/(1-rho_current))])])
            

            
            # Adaptive metropolis update for block 1
            log_am_lamb_block1 = log_am_lamb_block1 + step_sizes_1[mc_i]*(a_1 - self.a_target_1)
            #print(log_am_lamb_block1)
            mu_block1_update = mu_block1 + step_sizes_1[mc_i]*(update_block_1 - mu_block1)
            
            am_cov_block1 =  am_cov_block1 + \
                            step_sizes_1[mc_i]*( (update_block_1 - mu_block1).reshape(-1,1) @\
                                                        ((update_block_1 - mu_block1).reshape(-1,1).T) - am_cov_block1)
            
            mu_block1 = mu_block1_update
            
            # Store generated variables
            self.w_approx_store.append(w_approx_current)
            self.lambda_store.append(lambd_current)
            

            ############ BLOCK 2 Sampler #################
            #### Sample mu, gamma, sigma1 and sigma2  ####   
            am_lamb_block2 = np.exp(log_am_lamb_block2)
            proposal_vec2  = np.random.multivariate_normal(update_block_2,
                                                        am_lamb_block2*am_cov_block2+self.eps_2*np.eye(len(mu_block2)) )
            self.blk2_check.append((mu_block2, am_lamb_block2, am_cov_block2))

            mu_prop = proposal_vec2[0]
            gamma_prop = proposal_vec2[1]
            
            sigma_1_prop = np.exp(proposal_vec2[2])
            sigma_2_prop = np.exp(proposal_vec2[3])

            
            self.prop_check2.append((proposal_vec2, mu_block2))
            

            # Calc likelihood
            ll_prop = eval_ll(self.y_vals_true,
                            self.x_vals,
                            w_samples_1=w_approx_current[0:103],
                            w_samples_2=w_approx_current[103:],
                            sigma_1=sigma_1_prop,
                            sigma_2=sigma_2_prop,
                            tau_grid=self.tau_grid,
                            tau_grid_expanded=self.tau_grid_expanded,
                            mu=mu_prop,
                            gamma=gamma_prop,
                            base_quantile_mean=self.base_quantile_mean,
                            base_quantile_sd=self.base_quantile_sd,
                            base_quantile_v=self.base_quantile_v,
                            base_quantile_dist=self.base_quantile_dist)

            ll_curr = eval_ll(self.y_vals_true,
                            self.x_vals,
                            w_samples_1=w_approx_current[0:103],
                            w_samples_2=w_approx_current[103:],
                            sigma_1=sigma_1_current,
                            sigma_2=sigma_2_current,
                            tau_grid=self.tau_grid,
                            tau_grid_expanded=self.tau_grid_expanded,
                            mu=mu_current,
                            gamma=gamma_current,
                            base_quantile_mean=self.base_quantile_mean,
                            base_quantile_sd=self.base_quantile_sd,
                            base_quantile_v=self.base_quantile_v,
                            base_quantile_dist=self.base_quantile_dist)    

            self.ll_check_blk2.append((ll_prop, ll_curr))

            # Take metropolis step

            # Prior Probs
            log_prior_prop = student_t.logpdf(mu_prop, df=1,loc=0, scale=1) +\
                             student_t.logpdf(gamma_prop,df=1,loc=0, scale=1) +\
                            gamma.logpdf(sigma_1_prop**2,  a=2, scale=1/2) +\
                            gamma.logpdf(sigma_2_prop**2,  a=2, scale=1/2) 
             #norm.logpdf(mu_prop,0,10) +\
                            #norm.logpdf(gamma_prop,0,10) +\
            
            
            log_prior_current = student_t.logpdf(mu_current, df=1,loc=0, scale=1)+\
                                student_t.logpdf(gamma_current, df=1,loc=0, scale=1) +\
                                gamma.logpdf(sigma_1_current**2,  a=2, scale=1/2) +\
                                gamma.logpdf(sigma_2_current**2,  a=2, scale=1/2) 
            

            log_proposal_prop = 0  - sigma_1_prop  - sigma_2_prop
        
            
            current_vec2 = np.concatenate([np.array([mu_current]),
                                    np.array([gamma_current]),
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)])])
            

            log_proposal_current = 0 - sigma_1_current  - sigma_2_current
                                                            
            
            #if (ll_prop + log_prior_prop + log_proposal_current) > 0:
            #    break
                
            a_2 = np.exp(min(0,  (ll_prop + log_prior_prop + log_proposal_current) \
                            - (ll_curr +log_prior_current + log_proposal_prop) ))
            
            self.a_check_2.append((a_2, 
                            ll_prop + log_prior_prop + log_proposal_current,
                            ll_curr +log_prior_current + log_proposal_prop))
            
            #print(a)
            if np.random.uniform(0,1) < a_2:
                mu_current = mu_prop
                gamma_current = gamma_prop
                sigma_1_current = sigma_1_prop
                sigma_2_current = sigma_2_prop
                
                self.block_2_accept_cnts += 1
            else: 
                mu_current = mu_current
                gamma_current = gamma_current
                sigma_1_current = sigma_1_current
                sigma_2_current = sigma_2_current
            
            
            self.sigma_1_store.append(sigma_1_current)
            self.sigma_2_store.append(sigma_2_current)
            self.mu_store.append(mu_current)
            self.gamma_store.append(gamma_current)
            
            # Update AM block 2 sampling parameters
            update_block_2 = np.concatenate([np.array([mu_current]),
                                    np.array([gamma_current]),
                                    np.array([np.log(sigma_1_current),
                                                np.log(sigma_2_current)])])

            
            # Adaptive metropolis update for block 1
            log_am_lamb_block2 = log_am_lamb_block2 + step_sizes_2[mc_i]*(a_2 - self.a_target_2)
            #print(log_am_lamb_block1)
            mu_block2_update = mu_block2 + step_sizes_2[mc_i]*(update_block_2 - mu_block2)
            
            am_cov_block2 =  am_cov_block2 + \
                            step_sizes_2[mc_i]*( (update_block_2 - mu_block2).reshape(-1,1) @\
                                                        ((update_block_2 - mu_block2).reshape(-1,1).T) - am_cov_block2)
            
            mu_block2 = mu_block2_update.copy()
            self.cov_store_2.append(am_cov_block2)

            if ((mc_i%100 == 0) and (mc_i != 0)):
                e = time.time()
                print('Step: ', mc_i, ' Time Taken: ', e-s, 'Block 1 Accept: ', 100*self.block_1_accept_cnts/mc_i,' Block 2 Accept: ',100*self.block_2_accept_cnts/mc_i)
                s = time.time()
            
            if mc_i%1000 == 0:
                print('Lambda Current: ', lambd_current)
                print('Mu Current: ', mu_current)
                print('Gamma Current: ', gamma_current)
                print('Sigma 1 Current: ', sigma_1_current)
                print('Sigma 2 Current: ', sigma_2_current)

        
        output_dict = {'w': self.w_approx_store,
                       'lambda': self.lambda_store,
                        'mu': self.mu_store,
                         'gamma': self.gamma_store,
                         'sigma_1': self.sigma_1_store,
                         'sigma_2':self.sigma_2_store}
        
        return output_dict
    

