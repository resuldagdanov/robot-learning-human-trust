import torch

class Updater(object):
    def __init__(self,
                 policy_model,
                 reward_model,
                 configs):

        self.policy_model = policy_model
        self.reward_model = reward_model

        self.configs = configs
        self.device = configs.device

    def calculate_bc_loss(self,
                          predicted,
                          target):
        loss = torch.nn.MSELoss()(predicted,
                                  target)

        return loss
    
    def calculate_irl_loss(self,
                           demo_traj_reward,
                           robot_traj_reward,
                           probability,
                           nu_factor):
        loss = - torch.mean(demo_traj_reward) + \
            torch.log(torch.exp(nu_factor) * (torch.mean(torch.exp(robot_traj_reward) / (probability + 1e-7))))

        return loss
    
    def run_optimizers(self,
                       bc_loss,
                       irl_loss,
                       policy_optimizer,
                       reward_optimizer):

        policy_optimizer.zero_grad()
        bc_loss.backward()
        policy_optimizer.step()

        reward_optimizer.zero_grad()
        irl_loss.backward()
        reward_optimizer.step()

        return None
    
