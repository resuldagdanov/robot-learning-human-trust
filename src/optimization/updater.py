import torch

class Updater(object):

    def __init__(self,
                 configs,
                 policy_network):
        
        self.configs = configs
        self.device = configs.device

        self.policy_network = policy_network
        self.policy_optimizer = None

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

    def initialize_optimizers(self):
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                 lr=self.configs.policy_lr)
    
    def run_optimizers(self,
                       bc_loss):
        
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
    