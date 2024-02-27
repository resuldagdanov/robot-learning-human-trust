import numpy as np
import pandas as pd

from typing import Tuple

from scipy.stats import beta as beta_distribution
from scipy.optimize import minimize

from trusts.model_dynamics import TrustDistribution


class MLEOptimization(object):

    def __init__(self,
                 hil_experiment_data: pd.DataFrame,
                 epsilon_reward: float = 0.1) -> object:
        
        self.hil_experiment_data = hil_experiment_data
        self.epsilon_reward = epsilon_reward

        # min-max bounds for optimizing parameters
        self.bounds = [(1e-1, 1.0), # initial_alpha
                       (1e-1, 1.0), # initial_beta
                       (1e-1, 1e3), # w_success
                       (1e-1, 1e3), # w_failure
                       (1e-4, 1.0)] # gamma
    
    def negative_log_likelihood(self,
                                params: Tuple[float,
                                              float,
                                              float,
                                              float,
                                              float]) -> float:

        initial_alpha, initial_beta, w_success, w_failure, gamma = params
        
        self.trust_obj = TrustDistribution(initial_alpha=initial_alpha,
                                           initial_beta=initial_beta,
                                           gamma=gamma,
                                           initial_w_success=w_success,
                                           initial_w_failure=w_failure,
                                           epsilon_reward=self.epsilon_reward)
        
        # minimize sum of negative log likelihood
        log_likelihood = 0.0

        for _, row in self.hil_experiment_data.iterrows():

            trust_label = row["TrustLabel"]
            reward = row["Reward"]

            if np.isnan(trust_label) or np.isnan(reward):
                continue

            self.trust_obj.update_parameters(performance=reward)

            # check for invalid values of alpha and beta
            if np.isnan(self.trust_obj.alpha) or \
                np.isnan(self.trust_obj.beta) or \
                    self.trust_obj.alpha <= 0 or \
                        self.trust_obj.beta <= 0:
                # handle the case where alpha or beta becomes invalid
                continue
            
            else:
                log_likelihood += -np.log(beta_distribution.pdf(trust_label,
                                                                self.trust_obj.alpha,
                                                                self.trust_obj.beta))
        
        return log_likelihood
    
    def fit(self,
            initial_params: list) -> Tuple[TrustDistribution,
                                           float,
                                           float,
                                           float,
                                           float,
                                           float]:
        
        # initial guess for the parameters order:
        # initial_alpha, initial_beta, w_success, w_failure, gamma

        # use scipy minimize function to find the maximum likelihood estimation (MLE)
        result = minimize(fun=self.negative_log_likelihood,
                          x0=initial_params,
                          method="TNC",
                          bounds=self.bounds,
                          options={"xtol": 1e-10,
                                   "gtol": 1e-10,
                                   "maxiter": 7000})

        if not result.success:
            print("NOTE: Optimization Did Not converge! :")
            print(result.message)
        
        # extract the MLE parameters
        mle_initial_alpha, mle_initial_beta, mle_w_success, mle_w_failure, mle_gamma = result.x

        return self.trust_obj, mle_initial_alpha, mle_initial_beta, mle_w_success, mle_w_failure, mle_gamma
