from datetime import datetime


class Config(object):
    def __init__(self):

        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        time_info = current_date + "-" + current_time

        print("Current Time: ", time_info)
    
    def parameters(self):
        self.seed = 1234
        self.device = "cpu"

        self.batch_size = 32

        self.state_size = 3
        self.hidden_size = 64
        self.action_size = 3
        self.reward_size = 1


