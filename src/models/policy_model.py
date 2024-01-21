import torch


class RobotPolicy(torch.nn.Module):
    def __init__(self,
                 state_size=3,
                 hidden_size=64, 
                 out_size=3,
                 device="cpu"):
        
        super(RobotPolicy,
              self).__init__()

        self.device = device

        self.policy = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, out_size, bias=True))
        
    def forward(self,
                x):
        action = self.policy(x)
        
        return action

    def calculate_probability(self,
                              state_input):
        state_float = torch.FloatTensor(state_input)

        output = self.policy(state_float).detach()

        probability = torch.nn.functional.softmax(output,
                                                  dim=-1).numpy()
        
        return probability
