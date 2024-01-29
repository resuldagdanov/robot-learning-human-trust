# state vector normalization parameters
MAX_DISTANCE_TO_OBJECT = 1.5 # meters
MAX_DISTANCE_TO_TARGET = 1.5 # meters
MAX_DISTANCE_TO_START = 1.5 # meters
MAX_DISTANCE_TO_GROUND = 1.5 # meters

# action vector normalization parameters
END_EFFECTOR_POSITION_RANGE_X = [-1.5, 1.5] # meters
END_EFFECTOR_POSITION_RANGE_Y = [-1.5, 1.5] # meters
END_EFFECTOR_POSITION_RANGE_Z = [-1.5, 1.5] # meters

# behavior cloning policy model trajectories column names
COLUMN_NAMES = ["time_seconds",
                "distance_to_object",
                "distance_to_target",
                "distance_to_start",
                "distance_to_ground",
                "arm_action_x",
                "arm_action_y",
                "arm_action_z"]

# state vector column names
STATE_COLUMNS = ["distance_to_object",
                 "distance_to_target",
                 "distance_to_start",
                 "distance_to_ground"]

# action vector column names
ACTION_COLUMNS = ["arm_action_x",
                  "arm_action_y",
                  "arm_action_z"]

# number of state action pairs to discritize in each trajectory
TRAJECTORY_SIZE = 20 # equally spaced number of points

# collected dataset folder name in ("dataset \\ human_demonstrations \\ collection_date")
DEMO_COLLECTION_DATE = "2024_01_23" # year_month_day 

# number of training epoch for behavior cloning
BC_NUMBER_EPOCHS = 100
