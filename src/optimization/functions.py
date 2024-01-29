import os
import sys

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
