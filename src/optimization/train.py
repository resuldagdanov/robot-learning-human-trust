import os
import sys

import torch

from tqdm import tqdm
from typing import List

from updater import Updater

from torch.utils.tensorboard import SummaryWriter

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path
sys.path.append(parent_directory)

from utils import constants
from utils.config import Config
from utils.dataset_loader import PolicyDatasetLoader

from models.policy_model import RobotPolicy


def get_directories(configs: Config,
                    parent_directory: str) -> (List[str],
                                               str):
    
    grand_parent_path = os.path.dirname(parent_directory)
    results_path = os.path.join(grand_parent_path,
                                "results")
    
    policy_model_directory = os.path.join(results_path,
                                          "policy_network_params")
    if not os.path.exists(policy_model_directory):
        os.makedirs(policy_model_directory)
    
    policy_saving_path = configs.model_saving_path(directory=policy_model_directory)

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
    
    return json_paths, policy_saving_path


def save_policy(saving_path: str,
                loss_value_str: str) -> None:

    # save the action prediction model after each epoch
    filename = f"policy_network_epoch_{epoch + 1}_loss_{loss_value_str}.pt"
    torch.save(obj=policy_network.state_dict(),
               f=os.path.join(saving_path, filename))
    
    print(f"Saved Policy Network Model: {filename}")


if __name__ == "__main__":

    configs = Config()
    # call the parameters method to set the parameters
    configs.parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: ", device)
    configs.device = device

    # create and return preliminary base paths
    json_paths, policy_saving_path = get_directories(configs=configs,
                                                     parent_directory=parent_directory)
    
    # load demonstrations dataset
    training_data = PolicyDatasetLoader(demo_data_json_paths=json_paths)
    torch_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=configs.batch_size,
                                               shuffle=configs.data_shuffle,
                                               num_workers=configs.num_workers)

    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 std_min=configs.policy_std_min,
                                 std_max=configs.policy_std_max,
                                 device=configs.device)

    updater = Updater(configs=configs,
                      policy_network=policy_network)
    updater.initialize_optimizers()

    print("\n================== Initializing Training (chuff chuff) ==================\n")

    for epoch in range(constants.BC_NUMBER_EPOCHS):

        # loop through each batch inside the dataset
        for batch_data in tqdm(torch_loader):

            # get batch of data
            input_state = batch_data[0].float().to(device)
            output_action = batch_data[1].float().to(device)
            
            # forward pass to get mean of Gaussian distribution
            action_pred, action_std = policy_network.forward(x=input_state)
            action_prob, action_dist = policy_network.calculate_distribution(action_mu=action_pred,
                                                                             action_std=action_std)
            
            # policy distribution entropy
            entropy = action_dist.entropy()
            
            # compute negative log-likelihood loss value for maximum likelihood estimation
            loss_nll = - action_dist.log_prob(output_action).sum(axis=-1)
            batch_loss = loss_nll.mean()
            
            # backward pass and optimization
            updater.run_optimizers(bc_loss=batch_loss)
            
            loss_value = round(batch_loss.item(), 5)
            
        print(f"Epoch {epoch + 1}/{constants.BC_NUMBER_EPOCHS}, Batch Loss: {loss_value}")
            
        # save action policy network parameters after every epoch
        save_policy(saving_path=policy_saving_path,
                    loss_value_str=str(loss_value).replace(".", "_"))
