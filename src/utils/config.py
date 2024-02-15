import os

from datetime import datetime


class Config(object):

    def __init__(self) -> object:

        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        self.time_info = current_date + "-" + current_time

        print("Current Time: ", self.time_info)
    
    def parameters(self):

        self.seed = 1773
        self.device = "cpu"

        # neural network optimization parameters
        self.state_size = 3
        self.hidden_size = 16
        self.action_size = 3
        self.reward_size = 1

        # strategy for batch sampling
        self.batch_size = 128
        self.data_shuffle = True
        self.num_workers = 0

        # min-max log standard deviations for policy network output
        self.policy_log_std_min = -14 # exp(11.0) = 0.0
        self.policy_log_std_max = 1.4 # exp(1.1) = 4.0
        self.policy_log_std_init = 0.0 # exp(0.0) = 1.0
        
        # learning rates
        self.policy_lr = 1e-3
        self.reward_lr = 1e-4
        self.policy_scheduler_gamma = 0.9
        self.reward_scheduler_gamma = 0.9

        # 15% of the dataset will be used for validation
        self.validation_split = 0.15

        # patience for early stopping (number of epochs with no improvement on validation loss)
        self.early_stopping_patience = 20
    
    def model_saving_path(self,
                          directory: str) -> str:
        
        if not isinstance(directory, str):
            raise TypeError("Input 'directory' in model_saving_path function must be a string.")
        if not os.path.exists(directory):
            raise ValueError("Input 'directory' in model_saving_path function must be a valid path.")
        
        saving_path = os.path.join(directory, self.time_info)

        # create directory if does not exist
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        
        return saving_path
