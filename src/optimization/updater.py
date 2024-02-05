import torch

import numpy as np


class Updater(object):

    def __init__(self,
                 configs,
                 policy_network,
                 reward_network) -> object:
        
        self.configs = configs
        self.device = configs.device

        self.policy_network = policy_network
        self.reward_network = reward_network
        self.policy_optimizer = None
        self.reward_optimizer = None

    def calculate_bc_loss(self,
                          action_dist,
                          output_action) -> float:
        
        loss_nll = -action_dist.log_prob(output_action).sum(axis=-1)
        batch_loss = loss_nll.mean()

        return batch_loss
    
    def multivariate_gaussian_nll_loss(self,
                                       y_true: torch.FloatTensor,
                                       y_pred: torch.FloatTensor) -> float:
        
        # size of action space
        n_dims = int(int(y_pred.shape[1]) / 2)
        
        mu = y_pred[:, 0:n_dims]
        log_sigma = y_pred[:, n_dims:]
        
        # mean squared error given mean and standard deviation
        mse = 0.5 * torch.sum(torch.square((y_true - mu) / torch.exp(log_sigma)),
                               axis=1)
        # sum of predicted log standard deviations to penalize higher uncertainties in predictions
        sigma_trace = torch.sum(log_sigma,
                                 axis=1)
        # constant term related to the natural logarithm of 2 pi
        log_2_pi = 0.5 * n_dims * np.log(2 * np.pi)
        
        # multivariate Gaussian negative log-likelihood loss function
        log_likelihood = mse + sigma_trace # + log_2_pi

        return torch.mean(log_likelihood)
    
    def calculate_irl_loss(self,
                           demo_traj_reward,
                           robot_traj_reward,
                           probability,
                           nu_factor) -> float:
        
        loss = - torch.mean(demo_traj_reward) + \
            torch.log(torch.exp(nu_factor) * (torch.mean(torch.exp(robot_traj_reward) / (probability + 1e-7))))

        return loss

    def initialize_optimizers(self) -> None:
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                 lr=self.configs.policy_lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                 lr=self.configs.reward_lr)
    
    def run_policy_optimizer(self,
                             bc_loss) -> None:
        
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
    
    def run_reward_optimizer(self,
                             irl_loss) -> None:
        
        self.reward_optimizer.zero_grad()
        irl_loss.backward()
        self.reward_optimizer.step()
    