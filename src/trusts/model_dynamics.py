import numpy as np

from typing import List, Optional
from scipy import stats


class TrustDistribution(object):

    def __init__(self,
                 initial_alpha : float = 1.0,
                 initial_beta : float = 1.0,
                 gamma: float = 0.99,
                 initial_w_success: float = 1.0,
                 initial_w_failure: float = 1.0) -> object:
        
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.gamma = gamma
        self.w_success = initial_w_success
        self.w_failure = initial_w_failure
        
        self.list_alpha = [self.alpha]
        self.list_beta = [self.beta]
        self.list_w_success = [self.w_success]
        self.list_w_failure = [self.w_failure]

        self.update_beta_distribution()
    
    def update_parameters(self,
                          performance: float) -> None:
        
        # update only alpha parameter
        if performance > 0.0:

            # alpha parameter upgrade and extend the list of alpha parameters
            alpha_history = self.cumulative_value(self.list_alpha)
            self.alpha = alpha_history + self.w_success * performance
            self.list_alpha.append(self.alpha)

            # beta parameter will be the same as before and
            # no need to extend the list of beta parameters
            self.beta = self.cumulative_value(self.list_beta)

        # update only beta parameter
        else:

            # beta parameter upgrade and extend the list of beta parameters
            beta_history = self.cumulative_value(self.list_beta)
            self.beta = beta_history + self.w_failure * float(np.exp(abs(performance)))
            self.list_beta.append(self.beta)

            # alpha parameter will be the same as before and
            # no need to extend the list of alpha parameters
            self.alpha = self.cumulative_value(self.list_alpha)
        
        # update the underlying beta distribution based on current alpha and beta
        self.update_beta_distribution()
    
    def update_weights(self,
                       w_success: Optional[float] = None,
                       w_failure: Optional[float] = None) -> None:
        
        if w_success is not None:
            self.w_success = w_success
            self.list_w_success.append(w_success)
        
        if w_failure is not None:
            self.w_failure = w_failure
            self.list_w_failure.append(w_failure)
    
    def cumulative_value(self,
                         parameter_list: List[float]) -> None:
        
        num = len(parameter_list)

        history = 0.0
        for idx in range(num - 1, - 1, - 1):
            history += (self.gamma ** (num - 1 - idx)) * parameter_list[idx]
        
        return history
    
    def update_beta_distribution(self) -> None:

        self.distribution = stats.beta(self.alpha, self.beta)

    def get_trust_level(self) -> float:

        return self.distribution.mean()
    
    def get_beta_distribution_mean(self) -> float:

        return self.alpha / (self.alpha + self.beta)
