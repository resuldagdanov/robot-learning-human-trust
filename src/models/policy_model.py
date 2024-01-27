import torch


class RobotPolicy(torch.nn.Module):

    def __init__(self,
                 state_size: int = 4,
                 hidden_size: int = 64, 
                 out_size: int = 3,
                 std_min: float = -2.0,
                 std_max: float = 2.0,
                 device: str = "cpu"):
        
        super(RobotPolicy,
              self).__init__()
        
        self.std_min = std_min
        self.std_max = std_max
        self.device = device

        # policy network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                            torch.nn.ReLU())
        
        # action policy of squashed (with tanh) Gaussian distribution
        self.policy_mu = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                             torch.nn.Tanh())
        self.policy_log_std = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                                  torch.nn.ReLU())
    
    def forward(self,
                x: torch.Tensor) -> (torch.Tensor,
                                     torch.Tensor):
        
        # propagate through policy network backbone
        x = self.backbone(x)
        action_mu = self.policy_mu(x)
        action_log_std = self.policy_log_std(x)
        
        # constrain logits to reasonable values to match with demonstration distribution variance
        action_log_std = torch.clamp(input=action_log_std,
                                     min=self.std_min,
                                     max=self.std_max)
        action_std = torch.exp(action_log_std)
        
        # deterministic action resembling mean of Gaussian distribution
        return action_mu, action_std

    def calculate_distribution(self,
                               action_mu: torch.Tensor,
                               action_std: torch.Tensor) -> (torch.Tensor,
                                                             torch.distributions.Normal):
        
        action_distribution = torch.distributions.Normal(action_mu,
                                                         action_std)
        
        # log probability of given policy action
        probability = action_distribution.log_prob(value=action_mu)
        
        # reparameterization trick (comment this to output deterministic action)
        # sampled_action = action_distribution.rsample()
        
        return probability, action_distribution
