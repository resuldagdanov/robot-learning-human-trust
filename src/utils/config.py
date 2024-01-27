import os

from datetime import datetime


class Config(object):

    def __init__(self):

        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        self.time_info = current_date + "-" + current_time

        print("Current Time: ", self.time_info)
    
    def parameters(self):

        self.seed = 1234
        self.device = "cpu"

        self.batch_size = 2

        self.state_size = 4
        self.hidden_size = 64
        self.action_size = 3
        self.reward_size = 1

        self.data_shuffle = True
        self.num_workers = 0

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
