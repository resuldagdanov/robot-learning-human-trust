import torch


class RobotPolicy(torch.nn.Module):
    def __init__(self,
                 state_size=3,
                 hidden_size=64, 
                 out_size=1,
                 device="cpu"):
        
        super(RobotPolicy, self).__init__()

        self.device = device

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_size, bias=True),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        action = self.policy(x)
        
        return action
