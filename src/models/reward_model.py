import torch


class RewardFunction(torch.nn.Module):

    def __init__(self,
                 state_size: int = 3,
                 hidden_size: int = 64,
                 out_size: int = 1,
                 device: str = "cpu") -> object:
        
        super(RewardFunction,
              self).__init__()

        self.device = device
        
        # reward network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                            torch.nn.ReLU())
        
        # reward function output being a squashed (with Sigmoid) to result in a value between 0 and 1
        self.reward_value = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                                torch.nn.Sigmoid())
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        # propagate through reward network backbone
        x = self.backbone(x)
        value = self.reward_value(x)

        return value
    
    def estimate_reward(self,
                        state: torch.Tensor,
                        is_reward_inference: bool = False) -> torch.Tensor:
        
        if is_reward_inference:
            self.eval()

            with torch.no_grad():
                value = self.forward(x=state)
        
        else:
            value = self.forward(x=state)
        
        return value
