import torch


class RobotPolicy(torch.nn.Module):

    def __init__(self,
                 state_size: int = 4,
                 hidden_size: int = 64, 
                 out_size: int = 3,
                 device: str = "cpu"):
        
        super(RobotPolicy,
              self).__init__()

        self.device = device

        self.policy = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, out_size, bias=True),
                                          torch.nn.Tanh())
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        action = self.policy(x)
        
        return action

    def calculate_probability(self,
                              state_input):
        
        state_float = torch.FloatTensor(state_input)

        output = self.policy(state_float).detach()

        # apply sigmoid to map tanh output to [0, 1]
        probability = torch.nn.functional.sigmoid(output).numpy()
        
        return probability
