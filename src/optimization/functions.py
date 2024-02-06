import os
import sys

import numpy as np
import pandas as pd

import torch

from typing import List, Tuple, Union
from torch.utils.tensorboard import SummaryWriter

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path
sys.path.append(parent_directory)

from utils import common, constants
from utils.config import Config


def setup_config(device: torch.device) -> Config:

    if not isinstance(device, torch.device):
        raise TypeError("Input 'device' in setup_config function must be a torch.device.")
    
    configs = Config()
    # call the parameters method to set the parameters
    configs.parameters()

    configs.device = device

    torch.manual_seed(configs.seed)

    return configs


def get_directories(parent_directory: str,
                    data_folder_name: str = constants.DEMO_COLLECTION_DATE) -> (List[str],
                                                                                str):
    
    if not isinstance(parent_directory, str):
        raise TypeError("Input 'parent_directory' in get_directories function must be a string.")
    
    grand_parent_path = os.path.dirname(parent_directory)
    
    dataset_path = os.path.join(grand_parent_path,
                                "dataset")
    demo_path = os.path.join(dataset_path,
                             "human_demonstrations")
    dataset_folder = os.path.join(demo_path,
                                  data_folder_name)
    
    json_folder = os.path.join(dataset_folder,
                               "jsons")
    json_files = os.listdir(json_folder)
    json_paths = [os.path.join(json_folder, file)
                  for file in json_files if file.endswith(".json")]
    
    results_path = os.path.join(grand_parent_path,
                                "results")
    
    return json_paths, results_path


def create_directories(configs: Config,
                       results_path: str,
                       saving_policy: bool,
                       saving_reward: bool) -> Tuple[Union[str, None],
                                                     Union[str, None]]:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in create_directories function must be an object of Config() class.")
    if not isinstance(results_path, str):
        raise TypeError("Input 'results_path' in create_directories function must be a string.")
    if not isinstance(saving_policy, bool):
        raise TypeError("Input 'saving_policy' in create_directories function must be a boolean.")
    if not isinstance(saving_reward, bool):
        raise TypeError("Input 'saving_reward' in create_directories function must be a boolean.")
    
    policy_model_directory = os.path.join(results_path,
                                          "policy_network_params")
    reward_model_directory = os.path.join(results_path,
                                          "reward_network_params")
    
    if not os.path.exists(policy_model_directory):
        os.makedirs(policy_model_directory)
    if not os.path.exists(reward_model_directory):
        os.makedirs(reward_model_directory)
    
    if saving_policy:
        policy_saving_path = configs.model_saving_path(directory=policy_model_directory)
    else:
        policy_saving_path = None
    
    if saving_reward:
        reward_saving_path = configs.model_saving_path(directory=reward_model_directory)
    else:
        reward_saving_path = None

    return policy_saving_path, reward_saving_path


def load_policy_from_path(policy_network: torch.nn.Module,
                          results_path: str,
                          policy_loading_folder_name: str,
                          policy_params_name: str) -> torch.nn.Module:
    
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in load_policy_from_path function must be torch neural network module.")
    if not isinstance(results_path, str):
        raise TypeError("Input 'results_path' in load_policy_from_path function must be a string.")
    if not isinstance(policy_loading_folder_name, str):
        raise TypeError("Input 'policy_loading_folder_name' in load_policy_from_path function must be a string.")
    if not isinstance(policy_params_name, str):
        raise TypeError("Input 'policy_params_name' in load_policy_from_path function must be a string.")
    
    # location of the trained model parameters (make sure that the folder exists where model is trained priorly)
    policy_model_folder_path = os.path.join(results_path,
                                            "policy_network_params",
                                            policy_loading_folder_name)
    policy_model_path = os.path.join(policy_model_folder_path,
                                     policy_params_name)
    
    # set trained parameters to neural network
    policy_network = load_policy(policy_network=policy_network,
                                 model_path=policy_model_path)
    
    return policy_network


def save_policy(epoch: int,
                policy_network: torch.nn.Module,
                saving_path: str,
                loss_value_str: str) -> None:
    
    if not isinstance(epoch, int):
        raise TypeError("Input 'epoch' in save_policy function must be an integer.")
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in save_policy function must be torch neural network module.")
    if not isinstance(saving_path, str):
        raise TypeError("Input 'saving_path' in save_policy function must be a valid string path.")
    if not isinstance(loss_value_str, str):
        raise TypeError("Input 'loss_value_str' in save_policy function must be a string.")
    
    # save the action prediction model after each epoch
    filename = f"policy_network_epoch_{epoch + 1}_loss_{loss_value_str}.pt"
    torch.save(obj=policy_network.state_dict(),
               f=os.path.join(saving_path, filename))
    
    print(f"Saved Policy Network Model: {filename}")


def load_policy(policy_network: torch.nn.Module,
                model_path: str) -> torch.nn.Module:
    
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in load_policy function must be torch neural network module.")
    if not isinstance(model_path, str):
        raise TypeError("Input 'model_path' in load_policy function must be a valid string path.")
    
    policy_network.load_state_dict(torch.load(model_path))
    policy_network.eval()
    
    return policy_network


def load_reward_from_path(reward_network: torch.nn.Module,
                          results_path: str,
                          reward_loading_folder_name: str,
                          reward_params_name: str) -> torch.nn.Module:
    
    if not isinstance(reward_network, torch.nn.Module):
        raise TypeError("Input 'reward_network' in load_reward_from_path function must be torch neural network module.")
    if not isinstance(results_path, str):
        raise TypeError("Input 'results_path' in load_reward_from_path function must be a string.")
    if not isinstance(reward_loading_folder_name, str):
        raise TypeError("Input 'reward_loading_folder_name' in load_reward_from_path function must be a string.")
    if not isinstance(reward_params_name, str):
        raise TypeError("Input 'reward_params_name' in load_reward_from_path function must be a string.")
    
    # location of the trained model parameters (make sure that the folder exists where model is trained priorly)
    reward_model_folder_path = os.path.join(results_path,
                                            "reward_network_params",
                                            reward_loading_folder_name)
    reward_model_path = os.path.join(reward_model_folder_path,
                                     reward_params_name)
    
    # set trained parameters to neural network
    reward_network = load_reward(reward_network=reward_network,
                                 model_path=reward_model_path)
    
    return reward_network


def save_reward(epoch: int,
                reward_network: torch.nn.Module,
                saving_path: str,
                loss_value_str: str) -> None:
    
    if not isinstance(epoch, int):
        raise TypeError("Input 'epoch' in save_reward function must be an integer.")
    if not isinstance(reward_network, torch.nn.Module):
        raise TypeError("Input 'reward_network' in save_reward function must be torch neural network module.")
    if not isinstance(saving_path, str):
        raise TypeError("Input 'saving_path' in save_reward function must be a valid string path.")
    if not isinstance(loss_value_str, str):
        raise TypeError("Input 'loss_value_str' in save_reward function must be a string.")
    
    # save the action prediction model after each epoch
    filename = f"reward_network_epoch_{epoch + 1}_loss_{loss_value_str}.pt"
    torch.save(obj=reward_network.state_dict(),
               f=os.path.join(saving_path, filename))
    
    print(f"Saved Reward Network Model: {filename}")


def load_reward(reward_network: torch.nn.Module,
                model_path: str) -> torch.nn.Module:
    
    if not isinstance(reward_network, torch.nn.Module):
        raise TypeError("Input 'reward_network' in load_reward function must be torch neural network module.")
    if not isinstance(model_path, str):
        raise TypeError("Input 'model_path' in load_reward function must be a valid string path.")
    
    reward_network.load_state_dict(torch.load(model_path))
    reward_network.eval()
    
    return reward_network


def read_each_loader(configs: Config,
                     sample_data: Tuple[torch.Tensor,
                                        torch.Tensor,
                                        Union[torch.Tensor, int],
                                        Union[torch.Tensor, int]]) -> Tuple[torch.Tensor,
                                                                            torch.Tensor,
                                                                            Union[torch.Tensor, int],
                                                                            Union[torch.Tensor, int]]:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in read_each_loader function must be an instance of Config.")
    if not isinstance(sample_data, tuple) or len(sample_data) != 4:
        raise ValueError("Input 'sample_data' must be a tuple of length 4.")
    if not all(isinstance(elem, torch.Tensor) for elem in sample_data[:2]):
        raise TypeError("The first two elements in 'sample_data' must be instances of torch.Tensor.")
    
    # extract sample data in correct order
    input_state = sample_data[0].float().to(configs.device)
    output_action = sample_data[1].float().to(configs.device)
    trajectory_index = sample_data[2]
    state_number = sample_data[3]
    
    return input_state, output_action, trajectory_index, state_number


def convert_sample_2_df(input_state: torch.Tensor,
                        real_state_input: np.ndarray,
                        output_action: torch.Tensor,
                        real_action_output: np.ndarray,
                        action_log_prob: torch.Tensor,
                        action_pred: torch.Tensor,
                        action_std: torch.Tensor,
                        real_action_pred: np.ndarray,
                        trajectory_index: int,
                        state_number: int,
                        nll_loss: float) -> pd.DataFrame:
    
    if not isinstance(input_state, torch.Tensor):
        raise TypeError("Input 'input_state' in convert_sample_2_df function must be a torch.Tensor.")
    if not isinstance(real_state_input, np.ndarray):
        raise TypeError("Input 'real_state_input' in convert_sample_2_df function must be a numpy array.")
    if not isinstance(output_action, torch.Tensor):
        raise TypeError("Input 'output_action' in convert_sample_2_df function must be a torch.Tensor.")
    if not isinstance(real_action_output, np.ndarray):
        raise TypeError("Input 'real_action_output' in convert_sample_2_df function must be a numpy array.")
    if not isinstance(action_log_prob, torch.Tensor):
        raise TypeError("Input 'action_log_prob' in convert_sample_2_df function must be a torch.Tensor.")
    if not isinstance(action_pred, torch.Tensor):
        raise TypeError("Input 'action_pred' in convert_sample_2_df function must be a torch.Tensor.")
    if not isinstance(action_std, torch.Tensor):
        raise TypeError("Input 'action_std' in convert_sample_2_df function must be a torch.Tensor.")
    if not isinstance(real_action_pred, np.ndarray):
        raise TypeError("Input 'real_action_pred' in convert_sample_2_df function must be a numpy array.")
    if not isinstance(trajectory_index, int):
        raise TypeError("Input 'trajectory_index' in convert_sample_2_df function must be an integer.")
    if not isinstance(state_number, int):
        raise TypeError("Input 'state_number' in convert_sample_2_df function must be an integer.")
    if not isinstance(nll_loss, float):
        raise TypeError("Input 'nll_loss' in convert_sample_2_df function must be a float.")
    
    data = {}

    # add array elements to the dictionary
    data.update({constants.STATE_NORMALIZED_LABEL_NAME + f"_{i+1}": input_state.numpy()[i] for i in range(len(input_state))})
    data.update({constants.STATE_DENORMALIZED_LABEL_NAME + f"_{i+1}": real_state_input[i] for i in range(len(real_state_input))})
    data.update({constants.ACTION_NORMALIZED_LABEL_NAME + f"_{i+1}": output_action.numpy()[i] for i in range(len(output_action))})
    data.update({constants.ACTION_DENORMALIZED_LABEL_NAME + f"_{i+1}": real_action_output[i] for i in range(len(real_action_output))})
    data.update({constants.ACTION_PREDICTION_LOGPROB_NAME + f"_{i+1}": action_log_prob.detach().numpy()[i] for i in range(len(action_log_prob))})
    data.update({constants.ACTION_PREDICTION_NAME + f"_{i+1}": action_pred.detach().numpy()[i] for i in range(len(action_pred))})
    data.update({constants.ACTION_PREDICTION_STD_NAME + f"_{i+1}": action_std.detach().numpy()[i] for i in range(len(action_std))})
    data.update({constants.ACTION_PREDICTION_DENORMALIZED_NAME + f"_{i+1}": real_action_pred[i] for i in range(len(real_action_pred))})

    # add non-array elements to the dictionary
    data.update({
        constants.NUMBER_TRAJECTORY_COLUMN: trajectory_index,
        constants.STATE_NUMBER_COLUMN: state_number,
        constants.GAUSSIAN_NLL_LOSS_COLUMN: nll_loss
    })

    df = pd.DataFrame([data],
                      index=[state_number])
    
    return df


def extend_df_4_next_states(data_df: pd.DataFrame,
                            current_state_norm_estimation: torch.Tensor,
                            current_state_denorm_estimation: np.ndarray,
                            next_state_norm_label: np.ndarray,
                            next_state_denorm_label: np.ndarray,
                            next_state_norm_estimation: np.ndarray,
                            next_state_denorm_estimation: np.ndarray) -> pd.DataFrame:
    
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("Input 'data_df' in extend_df_4_next_state_estimation function must be a pandas dataframe.")
    if not isinstance(current_state_norm_estimation, torch.Tensor):
        raise TypeError("Input 'current_state_norm_estimation' in extend_df_4_next_state_estimation function must be a torch.Tensor.")
    if not isinstance(current_state_denorm_estimation, np.ndarray):
        raise TypeError("Input 'current_state_denorm_estimation' in extend_df_4_next_state_estimation function must be a numpy array.")
    if not isinstance(next_state_norm_label, np.ndarray):
        raise TypeError("Input 'next_state_norm_label' in extend_df_4_next_state_estimation function must be a numpy array.")
    if not isinstance(next_state_denorm_label, np.ndarray):
        raise TypeError("Input 'next_state_denorm_label' in extend_df_4_next_state_estimation function must be a numpy array.")
    if not isinstance(next_state_norm_estimation, np.ndarray):
        raise TypeError("Input 'next_state_norm_estimation' in extend_df_4_next_state_estimation function must be a numpy array.")    
    if not isinstance(next_state_denorm_estimation, np.ndarray):
        raise TypeError("Input 'next_state_denorm_estimation' in extend_df_4_next_state_estimation function must be a numpy array.")
    
    for i in range(len(current_state_norm_estimation)):
        data_df[constants.STATE_ESTIMATION_NORMALIZED_NAME + f"_{i+1}"] = current_state_norm_estimation.numpy()[i]
    for i in range(len(current_state_denorm_estimation)):
        data_df[constants.STATE_ESTIMATION_DENORMALIZED_NAME + f"_{i+1}"] = current_state_denorm_estimation[i]
    for i in range(len(next_state_norm_label)):
        data_df[constants.NEXT_STATE_NORMALIZED_LABEL_NAME + f"_{i+1}"] = next_state_norm_label[i]
    for i in range(len(next_state_denorm_label)):
        data_df[constants.NEXT_STATE_DENORMALIZED_LABEL_NAME + f"_{i+1}"] = next_state_denorm_label[i]
    for i in range(len(next_state_norm_estimation)):
        data_df[constants.NEXT_STATE_ESTIMATION_NORMALIZED_NAME + f"_{i+1}"] = next_state_norm_estimation[i]
    for i in range(len(next_state_denorm_estimation)):
        data_df[constants.NEXT_STATE_ESTIMATION_DENORMALIZED_NAME + f"_{i+1}"] = next_state_denorm_estimation[i]
    
    return data_df


def get_initial_position(data_loader: torch.utils.data.Dataset,
                         traj_start_index: int) -> np.ndarray:

    if not isinstance(data_loader, torch.utils.data.Dataset):
        raise TypeError("Input 'data_loader' in get_initial_position function must be a torch data loader object.")
    if not isinstance(traj_start_index, int):
        raise TypeError("Input 'traj_start_index' in get_initial_position function must be an integer.")
    
    # get the first sample in the dataset
    sample_data = data_loader[traj_start_index]

    # get initial end-effector position
    ee_location = common.denormalize_action(action_norm=sample_data[1].unsqueeze(0).numpy(),
                                            norm_range_list=data_loader.action_norms)[0]
    
    return ee_location


def calculate_next_state(action_denorm: np.ndarray,
                         obstacle_location: np.ndarray,
                         initial_state_location: np.ndarray,
                         target_location: np.ndarray) -> np.ndarray:
    
    if not isinstance(action_denorm, np.ndarray):
        raise TypeError("Input 'action_denorm' in calculate_next_state function must be a numpy array.")
    if not isinstance(obstacle_location, np.ndarray):
        raise TypeError("Input 'obstacle_location' in calculate_next_state function must be a numpy array.")
    if not isinstance(initial_state_location, np.ndarray):
        raise TypeError("Input 'initial_state_location' in calculate_next_state function must be a numpy array.")
    if not isinstance(target_location, np.ndarray):
        raise TypeError("Input 'target_location' in calculate_next_state function must be a numpy array.")
    if len(action_denorm) != len(constants.ACTION_COLUMNS):
        raise ValueError("The length of 'action_denorm' must be [action vector size].")
    
    # because both obstacle location and action prediction locations are computed w.r.t. the robot base,
    # we could direcly calculate euclidean distance between them without any transformation
    object_distance = np.linalg.norm(obstacle_location - action_denorm)
    target_distance = np.linalg.norm(target_location - action_denorm)
    start_distance = np.linalg.norm(initial_state_location - action_denorm)
    ground_distance = action_denorm[2] + constants.ROBOT_BASE_HEIGHT
    
    next_state_denorm = np.array([object_distance,
                                  target_distance,
                                  start_distance,
                                  ground_distance])
    
    return next_state_denorm


def trajectory_estimation(configs: Config,
                          updater_obj: object,
                          data_loader: torch.utils.data.Dataset,
                          policy_network: torch.nn.Module,
                          trajectory_length: int,
                          traj_start_index: int,
                          is_inference: bool=False) -> pd.DataFrame:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in trajectory_estimation function must be an instance of Config.")
    if not isinstance(updater_obj, object):
        raise TypeError("Input 'updater_obj' in trajectory_estimation function must be an object of Updater class.")
    if not isinstance(data_loader, torch.utils.data.Dataset):
        raise TypeError("Input 'data_loader' in trajectory_estimation function must be a torch data loader object.")
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in trajectory_estimation function must be a torch neural network module.")
    if not isinstance(trajectory_length, int):
        raise TypeError("Input 'trajectory_length' in trajectory_estimation function must be an integer.")
    if not isinstance(traj_start_index, int):
        raise TypeError("Input 'traj_start_index' in trajectory_estimation function must be an integer.")
    if not isinstance(is_inference, bool):
        raise TypeError("Input 'is_inference' in trajectory_estimation function must be a boolean.")
    
    # position (x, y, z w.r.t robot base) is constant throughout the trajectory
    initial_state_location = get_initial_position(data_loader=data_loader,
                                                  traj_start_index=traj_start_index)

    # initialize the state vector (normalized) from initial state value which is known
    state_0_idx = 0
    state_norm_estimation_vector = data_loader[traj_start_index + state_0_idx][0].unsqueeze(0).float().to(configs.device)

    # initialize the trajectory dataframe to store results
    created_trajectory_df = pd.DataFrame()

    # loop through the trajectory length
    for state_number in range(trajectory_length + constants.ACTION_LABEL_SHIFT_IDX):

        # actual state and action given the current state
        state_label_norm, action_label_norm, trajectory_index, _ = read_each_loader(configs=configs,
                                                                                    sample_data=tuple(data_loader[traj_start_index + state_number]))
        
        # estimate the action given the current state
        action_pred, action_std, action_log_prob, action_entropy, action_mu_and_std, action_dist = policy_network.estimate_action(state=state_norm_estimation_vector,
                                                                                                                                  is_inference=is_inference)
        nll_loss = updater_obj.multivariate_gaussian_nll_loss(y_true=action_label_norm,
                                                              y_pred=action_mu_and_std)
        
        # denormalize the state vector to get distances to object, target, start, and ground
        current_state_denorm_label = common.denormalize_state(state_norm=state_label_norm.numpy(),
                                                              norm_value_list=data_loader.state_norms)
        current_state_denorm_estimation = common.denormalize_state(state_norm=state_norm_estimation_vector.numpy(),
                                                                   norm_value_list=data_loader.state_norms)
        
        # denormalize the demonstration action to get actual x, y, z position of the end-effector
        action_denorm_label = common.denormalize_action(action_norm=action_label_norm.unsqueeze(0).detach().numpy(),
                                                        norm_range_list=data_loader.action_norms)
        
        # denormalize the action prediction to get x, y, z position of the end-effector
        action_denorm_prediction = common.denormalize_action(action_norm=action_pred.detach().numpy(),
                                                             norm_range_list=data_loader.action_norms)
        
        # x, y, z coordinates of the target location w.r.t robot base focal point is constant in this experiment
        target_location = np.array(constants.TARGET_LOCATION)

        # calculate the next denormalized actual state as given the current state and actual action
        next_state_denorm_label = calculate_next_state(action_denorm=action_denorm_label[0],
                                                       obstacle_location=np.array(constants.OBSTACLE_LOCATION),
                                                       initial_state_location=initial_state_location,
                                                       target_location=target_location)
        
        # normalize calculated actual next state
        next_state_norm_label = common.normalize_state(state=next_state_denorm_label,
                                                       norm_value_list=data_loader.state_norms)
        
        # calculate the next denormalized estimation state as given the current state and action prediction
        next_state_denorm_estimation = calculate_next_state(action_denorm=action_denorm_prediction[0],
                                                            obstacle_location=np.array(constants.OBSTACLE_LOCATION),
                                                            initial_state_location=initial_state_location,
                                                            target_location=target_location)
        
        # normalize calculated next state estimation
        next_state_norm_estimation = common.normalize_state(state=next_state_denorm_estimation,
                                                            norm_value_list=data_loader.state_norms)
        
        # convert the sample to a dataframe
        created_df = convert_sample_2_df(input_state=state_label_norm.squeeze(0),
                                         real_state_input=current_state_denorm_label,
                                         output_action=action_label_norm.squeeze(0),
                                         real_action_output=action_denorm_label[0],
                                         action_log_prob=action_log_prob.squeeze(0),
                                         action_pred=action_pred.squeeze(0),
                                         action_std=action_std.squeeze(0),
                                         real_action_pred=action_denorm_prediction[0],
                                         trajectory_index=int(trajectory_index),
                                         state_number=state_number,
                                         nll_loss=nll_loss.item())
        created_df = extend_df_4_next_states(data_df=created_df,
                                             current_state_norm_estimation=state_norm_estimation_vector.squeeze(0),
                                             current_state_denorm_estimation=current_state_denorm_estimation[0],
                                             next_state_norm_label=next_state_norm_label,
                                             next_state_denorm_label=next_state_denorm_label,
                                             next_state_norm_estimation=next_state_norm_estimation,
                                             next_state_denorm_estimation=next_state_denorm_estimation)
        
        # update the current state vector with estimated next state vector
        state_norm_estimation_vector = torch.from_numpy(next_state_norm_estimation).unsqueeze(0).float().to(configs.device)

        # append the sample dataframe to the trajectory dataframe
        created_trajectory_df = pd.concat([created_trajectory_df, created_df],
                                          ignore_index=True)
    
    return created_trajectory_df


def find_indices_of_trajectory_changes(dataset: torch.utils.data.Dataset) -> List[int]:

    if not isinstance(dataset, torch.utils.data.Dataset):
        raise TypeError("Input 'dataset' in find_indices_of_trajectory_changes function must be a torch data loader object.")
    
    indices_of_changes = []
    previous_value = None

    # second indices of each sample is the trajectory number (check dataset_loader.py __getitem__ for more info)
    for idx, sample in enumerate(dataset):
        current_value = sample[2]

        if previous_value is None or current_value == previous_value + 1:
            indices_of_changes.append(idx)

        previous_value = current_value

    return indices_of_changes


def get_estimated_rewards(configs: Config,
                          updater_obj: object,
                          data_loader: torch.utils.data.Dataset,
                          policy_network: torch.nn.Module,
                          reward_network: torch.nn.Module,
                          trajectory_indices: List[int],
                          traj_start_index: int,
                          is_inference_reward: bool) -> Tuple[pd.DataFrame,
                                                          torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in get_estimated_rewards function must be an instance of Config.")
    if not isinstance(updater_obj, object):
        raise TypeError("Input 'updater_obj' in get_estimated_rewards function must be an object of Updater class.")
    if not isinstance(data_loader, torch.utils.data.Dataset):
        raise TypeError("Input 'data_loader' in get_estimated_rewards function must be a torch data loader object.")
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in get_estimated_rewards function must be a torch neural network module.")
    if not isinstance(reward_network, torch.nn.Module):
        raise TypeError("Input 'reward_network' in get_estimated_rewards function must be a torch neural network module.")
    if not isinstance(trajectory_indices, list):
        raise TypeError("Input 'trajectory_indices' in get_estimated_rewards function must be a list.")
    if not isinstance(traj_start_index, int):
        raise TypeError("Input 'traj_start_index' in get_estimated_rewards function must be an integer.")
    if not isinstance(is_inference_reward, bool):
        raise TypeError("Input 'is_inference_reward' in get_estimated_rewards function must be a boolean.")
    
    # get trajectory dataframe with estimated state and actions
    data_traj_df = trajectory_estimation(configs=configs,
                                        updater_obj=updater_obj,
                                        data_loader=data_loader,
                                        policy_network=policy_network,
                                        trajectory_length=constants.TRAJECTORY_SIZE,
                                        traj_start_index=trajectory_indices[traj_start_index],
                                        is_inference=True)

    # extract estimated and actual state and action values from the trajectory dataframe
    norm_state_label_df = data_traj_df[[
        f"{constants.STATE_NORMALIZED_LABEL_NAME}_{i}" for i in range(1, len(constants.STATE_COLUMNS) + 1)]]
    norm_state_estim_df = data_traj_df[[
        f"{constants.STATE_ESTIMATION_NORMALIZED_NAME}_{i}" for i in range(1, len(constants.STATE_COLUMNS) + 1)]]
    norm_action_label_df = data_traj_df[[
        f"{constants.ACTION_NORMALIZED_LABEL_NAME}_{i}" for i in range(1, len(constants.ACTION_COLUMNS) + 1)]]
    norm_action_estim_df = data_traj_df[[
        f"{constants.ACTION_PREDICTION_NAME}_{i}" for i in range(1, len(constants.ACTION_COLUMNS) + 1)]]

    # get log probabilities of each predicted action in the trajectory and take average of log probabilities
    logprob_action_estim_df = data_traj_df[[
        f"{constants.ACTION_PREDICTION_LOGPROB_NAME}_{i}" for i in range(1, len(constants.ACTION_COLUMNS) + 1)]]
    logprob_action_estim_avg_df = logprob_action_estim_df.mean(axis=1)

    # concatenate states and actions in the trajectory for both estimations and demonstrations
    state_action_label_df = pd.concat([norm_state_label_df, norm_action_label_df],
                                      axis=1)
    state_action_estim_df = pd.concat([norm_state_estim_df, norm_action_estim_df],
                                      axis=1)
    
    # convert dataframes into tensor format to forward through neural network
    state_action_label_tensor = torch.tensor(state_action_label_df.values,
                                             dtype=torch.float64,
                                             device=configs.device)
    state_action_estim_tensor = torch.tensor(state_action_estim_df.values,
                                             dtype=torch.float64,
                                             device=configs.device)
    logprob_action_estim_avg_tensor = torch.tensor(logprob_action_estim_avg_df.values,
                                                   dtype=torch.float64,
                                                   device=configs.device).unsqueeze(1)
    
    # forward propagation through neural network
    reward_values_demonstration_data = reward_network.estimate_reward(state_action=state_action_label_tensor.float(),
                                                                      is_inference=is_inference_reward)
    reward_values_estimation_data = reward_network.estimate_reward(state_action=state_action_estim_tensor.float(),
                                                                   is_inference=is_inference_reward)
    
    return data_traj_df, reward_values_demonstration_data, reward_values_estimation_data, logprob_action_estim_avg_tensor


def trajectory_generation(configs: Config,
                          state_norms: List[float],
                          action_norms: List[float],
                          policy_network: torch.nn.Module,
                          average_initial_state_denorm: np.ndarray,
                          initial_state_location: np.ndarray) -> pd.DataFrame:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in trajectory_generation function must be an instance of Config.")
    if not isinstance(state_norms, list):
        raise TypeError("Input 'state_norms' in trajectory_generation function must be a list.")
    if not isinstance(action_norms, list):
        raise TypeError("Input 'action_norms' in trajectory_generation function must be a list.")
    if not isinstance(policy_network, torch.nn.Module):
        raise TypeError("Input 'policy_network' in trajectory_generation function must be a torch neural network module.")
    if not isinstance(average_initial_state_denorm, np.ndarray):
        raise TypeError("Input 'average_initial_state_denorm' in trajectory_generation function must be a numpy array.")
    if not isinstance(initial_state_location, np.ndarray):
        raise TypeError("Input 'initial_state_location' in trajectory_generation function must be a numpy array.")
    
    # normal random denormalized state vector of initial state
    variances = np.array([state_norms[0] / 8.0,
                          state_norms[1] / 8.0,
                          state_norms[2] / 8.0,
                          state_norms[3] / 8.0])
    initial_state_denorm = common.generate_gaussian_random_variables(means=average_initial_state_denorm,
                                                                     variances=variances)[0]
    input_state_norm = torch.from_numpy(common.normalize_state(state=initial_state_denorm,
                                                               norm_value_list=state_norms)).unsqueeze(0).float().to(configs.device)
    
    column_names = [constants.STATE_NUMBER_COLUMN]
    column_names += [constants.STATE_ESTIMATION_NORMALIZED_NAME + f"_{i+1}" for i in range(len(state_norms))]
    column_names += [constants.STATE_ESTIMATION_DENORMALIZED_NAME + f"_{i+1}" for i in range(len(state_norms))]
    column_names += [constants.ACTION_PREDICTION_NAME + f"_{i+1}" for i in range(len(action_norms))]
    column_names += [constants.ACTION_PREDICTION_DENORMALIZED_NAME + f"_{i+1}" for i in range(len(action_norms))]
    column_names += [constants.NEXT_STATE_ESTIMATION_NORMALIZED_NAME + f"_{i+1}" for i in range(len(state_norms))]
    column_names += [constants.NEXT_STATE_ESTIMATION_DENORMALIZED_NAME + f"_{i+1}" for i in range(len(state_norms))]
    
    created_trajectory_df = pd.DataFrame(index=range(constants.TRAJECTORY_SIZE),
                                         columns=column_names)

    for state_number in range(constants.TRAJECTORY_SIZE):
        output_action_norm, action_std, action_log_prob, action_entropy, action_mu_and_std, action_dist = policy_network.estimate_action(state=input_state_norm,
                                                                                                                                         is_inference=True)
        
        input_state_denorm = common.denormalize_state(state_norm=input_state_norm.numpy(),
                                                      norm_value_list=state_norms)
        output_action_denorm = common.denormalize_action(action_norm=output_action_norm.detach().numpy(),
                                                         norm_range_list=action_norms)
        
        next_state_denorm = calculate_next_state(action_denorm=output_action_denorm[0],
                                                 obstacle_location=np.array(constants.OBSTACLE_LOCATION),
                                                 initial_state_location=initial_state_location,
                                                 target_location=np.array(constants.TARGET_LOCATION))
        next_state_norm = common.normalize_state(state=next_state_denorm,
                                                 norm_value_list=state_norms)
        
        created_trajectory_df.loc[
            state_number, constants.STATE_NUMBER_COLUMN] = state_number
        created_trajectory_df.loc[
            state_number, constants.STATE_ESTIMATION_NORMALIZED_NAME + "_1" : constants.STATE_ESTIMATION_NORMALIZED_NAME + "_" + str(len(state_norms))] = input_state_norm.numpy()[0]
        created_trajectory_df.loc[
            state_number, constants.STATE_ESTIMATION_DENORMALIZED_NAME + "_1" : constants.STATE_ESTIMATION_DENORMALIZED_NAME + "_" + str(len(state_norms))] = input_state_denorm[0]
        created_trajectory_df.loc[
            state_number, constants.ACTION_PREDICTION_NAME + "_1" : constants.ACTION_PREDICTION_NAME + "_" + str(len(action_norms))] = output_action_norm.detach().numpy()[0]
        created_trajectory_df.loc[
            state_number, constants.ACTION_PREDICTION_DENORMALIZED_NAME + "_1" : constants.ACTION_PREDICTION_DENORMALIZED_NAME + "_" + str(len(action_norms))] = output_action_denorm[0]
        created_trajectory_df.loc[
            state_number, constants.NEXT_STATE_ESTIMATION_NORMALIZED_NAME + "_1" : constants.NEXT_STATE_ESTIMATION_NORMALIZED_NAME + "_" + str(len(state_norms))] = next_state_norm
        created_trajectory_df.loc[
            state_number, constants.NEXT_STATE_ESTIMATION_DENORMALIZED_NAME + "_1" : constants.NEXT_STATE_ESTIMATION_DENORMALIZED_NAME + "_" + str(len(state_norms))] = next_state_denorm
        
        input_state_norm = torch.from_numpy(next_state_norm).unsqueeze(0).float().to(configs.device)
    
    return created_trajectory_df
