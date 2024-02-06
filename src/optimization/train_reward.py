import os
import sys

import torch

from tqdm import tqdm

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path
sys.path.append(parent_directory)

from utils import constants
from utils.dataset_loader import PolicyDatasetLoader

from optimization.updater import Updater
from optimization.functions import setup_config, get_directories, create_directories, save_policy, load_policy_from_path
from optimization.functions import find_indices_of_trajectory_changes, get_estimated_rewards

from models.policy_model import RobotPolicy
from models.reward_model import RewardFunction


if __name__ == "__main__":

    # available training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: ", device)

    # setup hyperparameters
    configs = setup_config(device=device)

    # create and return preliminary base paths
    json_paths_train, results_path = get_directories(parent_directory=parent_directory,
                                                     data_folder_name=constants.DEMO_COLLECTION_DATE)
    json_paths_validate, _ = get_directories(parent_directory=parent_directory,
                                             data_folder_name=constants.TEST_COLLECTION_DATE)
    
    # load train-validation dataset of demonstrations
    all_train_data = PolicyDatasetLoader(demo_data_json_paths=json_paths_train)
    all_validate_data = PolicyDatasetLoader(demo_data_json_paths=json_paths_validate)

    # get all indice numbers where the new trajectory is initialized in the dataset
    trajectory_indices_train = find_indices_of_trajectory_changes(dataset=all_train_data)
    trajectory_indices_valid = find_indices_of_trajectory_changes(dataset=all_validate_data)

    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 log_std_min=configs.policy_log_std_min,
                                 log_std_max=configs.policy_log_std_max,
                                 log_std_init=configs.policy_log_std_init,
                                 device=configs.device)
    
    reward_network = RewardFunction(state_action_size=configs.state_action_size,
                                    hidden_size=configs.hidden_size,
                                    out_size=configs.reward_size,
                                    device=configs.device)
    
    updater_obj = Updater(configs=configs,
                          policy_network=policy_network,
                          reward_network=reward_network)
    updater_obj.initialize_optimizers()

    # folder name where policy model parameters are located ("results / policy_network_params / loading_folder_name")
    policy_loading_folder_name = constants.POLICY_LOADING_FOLDER
    policy_params_name = constants.POLICY_PARAMS_NAME

    # load pretrained policy network parameters
    policy_network = load_policy_from_path(policy_network=policy_network,
                                           results_path=results_path,
                                           policy_loading_folder_name=policy_loading_folder_name,
                                           policy_params_name=policy_params_name)
    
    # set model to evaluation mode
    for param in policy_network.parameters():
        param.requires_grad = False
    policy_network = policy_network.eval()

    # parameters for early stopping criteria
    best_bc_val_loss = float("inf")
    early_stopping_counter = 0

    # currently nu weight is zero; will be updated later
    nu_factor = torch.tensor(0.0)

    print("\n================== Training Initialized (chuff chuff) ==================\n")

    for epoch in range(constants.RF_NUMBER_EPOCHS):

        print("================== Training Phase ==================")
        reward_network.train()
        cummulative_train_loss = 0.0

        # loop through each separate trajectory inside the training dataset
        for traj_start_index_train in trajectory_indices_train:

            traj_df, reward_values_demo_data, reward_values_estim_data, logprob_action_estim_avg = get_estimated_rewards(configs=configs,
                                                                                                                         updater_obj=updater_obj,
                                                                                                                         data_loader=all_train_data,
                                                                                                                         policy_network=policy_network,
                                                                                                                         reward_network=reward_network,
                                                                                                                         trajectory_indices=trajectory_indices_train,
                                                                                                                         traj_start_index=traj_start_index_train)
            
            # calculate irl loss function value of the particilar trajectories
            irl_train_loss = updater_obj.calculate_irl_loss(demo_traj_reward=reward_values_demo_data,
                                                            robot_traj_reward=reward_values_estim_data,
                                                            log_probability=logprob_action_estim_avg,
                                                            nu_factor=nu_factor)
            
            # backward pass and optimization
            updater_obj.run_reward_optimizer(irl_loss=irl_train_loss)

            cummulative_train_loss += irl_train_loss.item()
        
        # calculate average training loss in the current epoch
        avg_bc_train_loss_value = round(cummulative_train_loss / len(trajectory_indices_train), 5)
        