# state vector normalization parameters
MAX_DISTANCE_TO_OBJECT = 2.0 # meters
MAX_DISTANCE_TO_TARGET = 2.0 # meters
MAX_DISTANCE_TO_START = 2.0 # meters
MAX_DISTANCE_TO_GROUND = 2.0 # meters

# action vector normalization parameters
END_EFFECTOR_POSITION_RANGE_X = [-2.0, 2.0] # meters
END_EFFECTOR_POSITION_RANGE_Y = [-2.0, 2.0] # meters
END_EFFECTOR_POSITION_RANGE_Z = [-2.0, 2.0] # meters

# state vector column names
STATE_COLUMNS = ["distance_to_object",
                 "distance_to_target",
                 "distance_to_start",
                 "distance_to_ground"]

# action vector column names
ACTION_COLUMNS = ["arm_action_x",
                  "arm_action_y",
                  "arm_action_z"]

# behavior cloning policy model trajectories column names
COLUMN_NAMES = ["time_seconds"] + \
                STATE_COLUMNS + \
                ACTION_COLUMNS

# column names for normalized and denormalized state and action vectors
STATE_NORMALIZED_LABEL_NAME = "state_label_norm" # normalized state vector
STATE_DENORMALIZED_LABEL_NAME = "state_label_denorm" # denormalized state vector
STATE_ESTIMATION_NORMALIZED_NAME = "state_est_norm" # estimated normalized state vector
STATE_ESTIMATION_DENORMALIZED_NAME = "state_est_denorm" # estimated denormalized state vector
NEXT_STATE_NORMALIZED_LABEL_NAME = "next_state_label_norm" # normalized next state vector
NEXT_STATE_DENORMALIZED_LABEL_NAME = "next_state_label_denorm" # denormalized next state vector
NEXT_STATE_ESTIMATION_NORMALIZED_NAME = "next_state_est_norm" # estimated normalized state vector
NEXT_STATE_ESTIMATION_DENORMALIZED_NAME = "next_state_est_denorm" # estimated denormalized state vector
ACTION_NORMALIZED_LABEL_NAME = "action_label_norm" # normalized action label
ACTION_DENORMALIZED_LABEL_NAME = "action_label_denorm" # denormalized action vector
ACTION_PREDICTION_NAME = "action_pred_norm" # mean of distribution
ACTION_PREDICTION_DENORMALIZED_NAME = "action_pred_denorm" # x, y, z locations
ACTION_PREDICTION_STD_NAME = "action_pred_std" # standard deviation
ACTION_PREDICTION_LOGPROB_NAME = "action_pred_logprob" # log probability of action prediction
ACTION_PREDICTION_ENTROPY_NAME = "action_pred_entropy" # entropy of action prediction
ACTION_LABEL_LOGPROB_NAME = "action_label_logprob" # log probability of demonstration action

# column name to represent number index of the state in a trajectory
STATE_NUMBER_COLUMN = "state_number"

# trajectory index number column name
NUMBER_TRAJECTORY_COLUMN = "trajectory_index"

# multivarate gaussian log likelihood loss value column name
GAUSSIAN_NLL_LOSS_COLUMN = "gnll_loss"

# end-effector position is next action value for the given state (constant : -1)
ACTION_LABEL_SHIFT_IDX = -1

# number of state action pairs to discritize in each trajectory
TRAJECTORY_SIZE = 30 # equally spaced number of points

# collected dataset folder name in ("dataset \\ human_demonstrations \\ collection_date")
DEMO_COLLECTION_DATE = "2024_01_23" # year_month_day NOTE: make sure that data date is correct
TEST_COLLECTION_DATE = "2024_02_02_Test" # year_month_day NOTE: make sure folder name is correct

# experiment obstacle location x, y, z coordinates
OBSTACLE_LOCATION = [-0.4, 0.0, 0.554] # meters NOTE: make sure the alignment with experiment

# altitute of the base of the robot from the ground as a focal point
ROBOT_BASE_HEIGHT = 0.1685 # meters NOTE: make sure the alignment with robot experiment

# experiment target location x, y, z coordinates
TARGET_LOCATION = [-0.36, 0.77, -ROBOT_BASE_HEIGHT] # meters NOTE: make sure the alignment with robot experiment

# number of training epoch for behavior cloning
BC_NUMBER_EPOCHS = 100
