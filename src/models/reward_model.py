import torch


class RewardFunction(torch.nn.Module):
    def __init__(self,
                 state_action_size=6,
                 hidden_size=64, 
                 out_size=1,
                 device="cpu"):
        
        super(RewardFunction,
              self).__init__()

        self.device = device

        self.reward = torch.nn.Sequential(torch.nn.Linear(state_action_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, out_size, bias=True),
                                          torch.nn.Tanh())
        
    def forward(self, x):
        value = self.reward(x)

        return value
