import os
import sys

import numpy as np
import pandas as pd

import torch

from typing import List
from torch.utils.tensorboard import SummaryWriter

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path
sys.path.append(parent_directory)

from utils import constants
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


def get_directories(parent_directory: str) -> (List[str],
                                               str):
    
    if not isinstance(parent_directory, str):
        raise TypeError("Input 'parent_directory' in get_directories function must be a string.")
    
    grand_parent_path = os.path.dirname(parent_directory)
    
    dataset_path = os.path.join(grand_parent_path,
                                "dataset")
    demo_path = os.path.join(dataset_path,
                             "human_demonstrations")
    dataset_folder = os.path.join(demo_path,
                                  constants.DEMO_COLLECTION_DATE)
    
    json_folder = os.path.join(dataset_folder,
                               "jsons")
    json_files = os.listdir(json_folder)
    json_paths = [os.path.join(json_folder, file)
                  for file in json_files if file.endswith(".json")]
    
    results_path = os.path.join(grand_parent_path,
                                "results")
    
    return json_paths, results_path


def create_directories(configs: Config,
                       results_path: str) -> str:
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in create_directories function must be an object of Config() class.")
    if not isinstance(results_path, str):
        raise TypeError("Input 'parent_directory' in create_directories function must be a string.")
    
    policy_model_directory = os.path.join(results_path,
                                          "policy_network_params")
    if not os.path.exists(policy_model_directory):
        os.makedirs(policy_model_directory)
    
    policy_saving_path = configs.model_saving_path(directory=policy_model_directory)

    return policy_saving_path


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


def read_each_loader(configs: Config,
                     sample_data: tuple) -> (torch.Tensor,
                                             torch.Tensor,
                                             int,
                                             int):
    
    if not isinstance(configs, Config):
        raise TypeError("Input 'configs' in read_each_loader function must be an instance of Config.")
    if not isinstance(sample_data, tuple):
        raise TypeError("Input 'sample_data' in read_each_loader function must be a tuple.")
    if len(sample_data) != 4:
        raise ValueError("Input 'sample_data' must be a tuple of length 4.")
    
    # extract sample data in correct order
    input_state = sample_data[0].float().to(configs.device)
    output_action = sample_data[1].float().to(configs.device)
    trajectory_index = sample_data[2]
    state_number = sample_data[3]
    
    return input_state, output_action, trajectory_index, state_number


def convert_sample_2_df(input_state: torch.Tensor,
                        real_state_input: np.array,
                        output_action: torch.Tensor,
                        real_action_output: np.array,
                        action_log_prob: torch.Tensor,
                        action_pred: torch.Tensor,
                        action_std: torch.Tensor,
                        real_action_pred: np.array,
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
    data.update({f"input_state_{i+1}": input_state.numpy()[i] for i in range(input_state.shape[0])})
    data.update({f"real_state_input_{i+1}": real_state_input[i] for i in range(len(real_state_input))})
    data.update({f"output_action_{i+1}": output_action.numpy()[i] for i in range(output_action.shape[0])})
    data.update({f"real_action_output_{i+1}": real_action_output[i] for i in range(len(real_action_output))})
    data.update({f"action_log_prob_{i+1}": action_log_prob[0].detach().numpy()[i] for i in range(action_log_prob.shape[1])})
    data.update({f"action_pred_{i+1}": action_pred[0].detach().numpy()[i] for i in range(action_pred.shape[1])})
    data.update({f"action_std_{i+1}": action_std[0].detach().numpy()[i] for i in range(action_std.shape[1])})
    data.update({f"real_action_pred_{i+1}": real_action_pred[i] for i in range(len(real_action_pred))})

    # add non-array elements to the dictionary
    data.update({
        "trajectory_index": trajectory_index,
        "state_number": state_number,
        "nll_loss": nll_loss
    })

    df = pd.DataFrame([data],
                      index=[state_number])
    
    return df
