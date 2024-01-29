import torch

import numpy as np


class Updater(object):

    def __init__(self,
                 configs,
                 policy_network) -> object:
        
        self.configs = configs
        self.device = configs.device

        self.policy_network = policy_network
        self.policy_optimizer = None

    def calculate_bc_loss(self,
                          action_dist,
                          output_action) -> float:
        
        loss_nll = -action_dist.log_prob(output_action).sum(axis=-1)
        batch_loss = loss_nll.mean()

        return batch_loss
    
    def gaussian_nll_loss(self,
                          y_true: torch.FloatTensor,
                          y_pred: torch.FloatTensor) -> float:
        
        n_dims = int(int(y_pred.shape[1]) / 2)
        
        mu = y_pred[:, 0:n_dims]
        log_sigma = y_pred[:, n_dims:]
        
        mse = -0.5 * torch.sum(torch.square((y_true - mu) / torch.exp(log_sigma)),
                               axis=1)
        sigma_trace = -torch.sum(log_sigma,
                                 axis=1)
        log_2_pi = -0.5 * n_dims * np.log(2 * np.pi)
        
        log_likelihood = mse + sigma_trace + log_2_pi

        return torch.mean(-log_likelihood)
    
    def calculate_irl_loss(self,
                           demo_traj_reward,
                           robot_traj_reward,
                           probability,
                           nu_factor):
        
        loss = - torch.mean(demo_traj_reward) + \
            torch.log(torch.exp(nu_factor) * (torch.mean(torch.exp(robot_traj_reward) / (probability + 1e-7))))

        return loss

    def initialize_optimizers(self) -> None:
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                 lr=self.configs.policy_lr)
    
    def run_optimizers(self,
                       bc_loss):
        
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
    