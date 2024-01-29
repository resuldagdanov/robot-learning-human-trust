import torch


class RobotPolicy(torch.nn.Module):

    def __init__(self,
                 state_size: int = 4,
                 hidden_size: int = 64, 
                 out_size: int = 3,
                 log_std_min: float = -11,
                 log_std_max: float = 1.1,
                 log_std_init: float = 0.0,
                 device: str = "cpu") -> object:
        
        super(RobotPolicy,
              self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        # policy network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                            torch.nn.ReLU())
        
        # action policy of squashed (with tanh) Gaussian distribution
        self.policy_mu = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                             torch.nn.Tanh())
        
        # log standard deviation is the same size as action vector
        self.policy_log_std = torch.nn.Parameter(torch.full((1, out_size),
                                                            float(log_std_init)))
        self.apply(self.init_weights)
    
    def forward(self,
                x: torch.Tensor) -> (torch.Tensor,
                                     torch.Tensor):
        
        # propagate through policy network backbone
        x = self.backbone(x)
        action_mu = self.policy_mu(x)

        # action_log_std = self.policy_log_std(x)
        action_log_std = self.policy_log_std.expand_as(action_mu)
        
        # constrain logits to reasonable values to match with demonstration distribution variance
        action_log_std = torch.clamp(input=action_log_std,
                                     min=self.log_std_min,
                                     max=self.log_std_max)
        action_std = torch.exp(action_log_std)
        
        # deterministic action resembling mean of Gaussian distribution
        return action_mu, action_std

    def calculate_distribution(self,
                               action_mu: torch.Tensor,
                               action_std: torch.Tensor) -> (torch.Tensor,
                                                             torch.distributions.Normal):
        
        # action distibution is assumed to be Gaussian with mean and std
        action_distribution = torch.distributions.Normal(action_mu,
                                                         action_std)
        
        # log probability of given policy action
        log_probability = action_distribution.log_prob(value=action_mu)

        # reparameterization trick (comment this to output deterministic action)
        # sampled_action = action_distribution.rsample()
        
        return log_probability, action_distribution
    
    def init_weights(self,
                     hidden_layer: torch.nn.Linear,
                     gain: float=1.0) -> None:
        
        # xavier_normal_ is used when activation functions is tanh or sigmoid
        # orthogonal_ is used when activation functions is relu
        if isinstance(hidden_layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(hidden_layer.weight,
                                         gain=gain)
            hidden_layer.bias.data.fill_(0.0)
        else:
            pass
    