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
from optimization.functions import setup_config, get_directories, create_directories, save_policy

from models.policy_model import RobotPolicy


if __name__ == "__main__":

    # available training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: ", device)

    # setup hyperparameters
    configs = setup_config(device=device)

    # create and return preliminary base paths
    json_paths, results_path = get_directories(parent_directory=parent_directory)
    policy_saving_path = create_directories(configs=configs,
                                            results_path=results_path)
    
    # load demonstrations dataset
    training_data = PolicyDatasetLoader(demo_data_json_paths=json_paths)
    torch_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=configs.batch_size,
                                               shuffle=configs.data_shuffle,
                                               num_workers=configs.num_workers)

    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 log_std_min=configs.policy_log_std_min,
                                 log_std_max=configs.policy_log_std_max,
                                 log_std_init=configs.policy_log_std_init,
                                 device=configs.device)

    updater = Updater(configs=configs,
                      policy_network=policy_network)
    updater.initialize_optimizers()

    print("\n================== Initializing Training (chuff chuff) ==================\n")

    for epoch in range(constants.BC_NUMBER_EPOCHS):

        # loop through each batch inside the dataset
        for batch_data in tqdm(torch_loader):
            
            # get batch of data
            input_state = batch_data[0].float().to(configs.device)
            output_action = batch_data[1].float().to(configs.device)

            # forward pass to get mean of Gaussian distribution
            action_pred, action_std = policy_network.forward(x=input_state)
            action_log_prob, action_dist = policy_network.calculate_distribution(action_mu=action_pred,
                                                                                 action_std=action_std)
            
            # policy distribution entropy
            entropy = action_dist.entropy()
            
            action_mu_and_std = torch.cat((action_pred, action_std),
                                          dim=-1)
            
            # multivariate Gaussian negative log-likelihood loss function
            nll_loss = updater.gaussian_nll_loss(y_true=output_action,
                                                 y_pred=action_mu_and_std)
            
            # backward pass and optimization
            updater.run_optimizers(bc_loss=nll_loss)
            
            bc_loss_value = round(nll_loss.item(), 5)
            
        print(f"Epoch {epoch + 1}/{constants.BC_NUMBER_EPOCHS}, Batch Loss: {bc_loss_value}")
            
        # save action policy network parameters after every epoch
        save_policy(epoch=epoch,
                    policy_network=policy_network,
                    saving_path=policy_saving_path,
                    loss_value_str=str(bc_loss_value).replace(".", "_"))
