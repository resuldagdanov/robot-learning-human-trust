import torch

import numpy as np


class Updater(object):

    def __init__(self,
                 configs: object,
                 policy_network: torch.nn.Module,
                 reward_network: torch.nn.Module) -> object:
        
        self.configs = configs
        self.device = configs.device

        self.policy_network = policy_network
        self.reward_network = reward_network
        self.policy_optimizer = None
        self.reward_optimizer = None

    def calculate_bc_loss(self,
                          action_dist: torch.Tensor,
                          output_action: torch.Tensor) -> float:
        
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
                           demo_traj_reward: torch.Tensor,
                           robot_traj_reward: torch.Tensor,
                           log_probability: torch.Tensor,
                           nu_factor: torch.Tensor) -> float:
        
        # max-entropy inverse reinforcement learning loss function (similar to guided cost learning)
        loss = -torch.mean(demo_traj_reward) + \
                    (torch.exp(nu_factor) * \
                        (torch.logsumexp(robot_traj_reward - log_probability,
                                         dim=0,
                                         keepdim=True)) - \
                        torch.log(torch.Tensor([len(robot_traj_reward)]))
                    )

        return loss
    
    def calculate_sample_traj_loss(self,
                                   nu_factor: torch.Tensor,
                                   robot_traj_reward: torch.Tensor,
                                   log_probability: torch.Tensor) -> float:
        
        # loss value estimated from approximation of the background distribution (Z) in max-entropy IRL
        loss = torch.exp(nu_factor) * \
            (
                torch.logsumexp(robot_traj_reward - log_probability,
                                dim=0,
                                keepdim=True)) - \
                torch.log(torch.Tensor([len(robot_traj_reward)])
            )
        
        return loss

    def initialize_optimizers(self) -> None:
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                 lr=self.configs.policy_lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                 lr=self.configs.reward_lr)
    
    def run_policy_optimizer(self,
                             bc_loss: torch.Tensor) -> None:
        
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
    
    def run_reward_optimizer(self,
                             irl_loss: torch.Tensor) -> None:
        
        self.reward_optimizer.zero_grad()
        irl_loss.backward()
        self.reward_optimizer.step()
    