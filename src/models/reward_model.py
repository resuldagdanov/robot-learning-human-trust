import torch


class RewardFunction(torch.nn.Module):

    def __init__(self,
                 state_action_size: int = 7,
                 hidden_size: int = 64, 
                 out_size: int = 1,
                 device: str = "cpu") -> object:
        
        super(RewardFunction,
              self).__init__()

        self.device = device

        # reward network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_action_size, hidden_size, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                            torch.nn.ReLU())
        
        # reward function output being a squashed (with Tanh) to result in a value between -1 and -1
        self.reward = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                          torch.nn.Tanh())
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        # propagate through reward network backbone
        x = self.backbone(x)
        value = self.reward(x)

        return value
    
    def estimate_reward(self,
                        state_action: torch.Tensor,
                        is_inference: bool = False) -> torch.Tensor:
        
        if is_inference:
            self.eval()
            # inference
            with torch.no_grad():
                reward_value = self.forward(x=state_action)
        else:
            reward_value = self.forward(x=state_action)
        
        return reward_value
