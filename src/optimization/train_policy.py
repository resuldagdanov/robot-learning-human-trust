import os
import sys

import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path
sys.path.append(parent_directory)

from utils import constants
from utils.dataset_loader import PolicyDatasetLoader

from optimization.updater import Updater
from optimization.functions import setup_config, get_directories, create_directories, save_policy, read_each_loader

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
    all_data = PolicyDatasetLoader(demo_data_json_paths=json_paths)

    # split dataset into training and validation sets
    train_data, val_data = train_test_split(all_data,
                                            test_size=configs.validation_split,
                                            random_state=configs.seed)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=configs.batch_size,
                                               shuffle=configs.data_shuffle,
                                               num_workers=configs.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=configs.batch_size,
                                             shuffle=False,
                                             num_workers=configs.num_workers)
    
    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 log_std_min=configs.policy_log_std_min,
                                 log_std_max=configs.policy_log_std_max,
                                 log_std_init=configs.policy_log_std_init,
                                 device=configs.device)

    updater = Updater(configs=configs,
                      policy_network=policy_network,
                      reward_network=None)
    updater.initialize_optimizers()

    # parameters for early stopping criteria
    best_bc_val_loss = float("inf")
    early_stopping_counter = 0

    print("\n================== Training Initialized (chuff chuff) ==================\n")

    for epoch in range(constants.BC_NUMBER_EPOCHS):

        print("================== Training Phase ==================")
        policy_network.train()
        cummulative_train_loss = 0.0

        # loop through each batch inside the training dataset
        for batch_train_data in tqdm(train_loader):

            # get batch of data
            input_state, output_action, _, _ = read_each_loader(configs=configs,
                                                                sample_data=tuple(batch_train_data))
            
            # forward pass to get Gaussian distribution
            action_pred, action_std, action_log_prob, action_entropy, action_mu_and_std, action_dist = policy_network.estimate_action(state=input_state,
                                                                                                                                      is_inference=False)

            # multivariate Gaussian negative log-likelihood loss function
            nll_train_loss = updater.multivariate_gaussian_nll_loss(y_true=output_action,
                                                                    y_pred=action_mu_and_std)
            
            # backward pass and optimization
            updater.run_policy_optimizer(bc_loss=nll_train_loss)

            cummulative_train_loss += nll_train_loss.item()
        
        # calculate average training loss in the current epoch
        avg_bc_train_loss_value = round(cummulative_train_loss / len(train_loader), 5)
        
        print("================== Validation Phase ==================")
        policy_network.eval()
        cummulative_val_loss = 0.0

        # freeze neural network parameters during validation
        with torch.no_grad():
            # loop through each batch inside the validation dataset
            for batch_val_data in tqdm(val_loader):
                input_state, output_action, _, _ = read_each_loader(configs=configs,
                                                                    sample_data=tuple(batch_val_data))
                action_pred, action_std, action_log_prob, action_entropy, action_mu_and_std, action_dist = policy_network.estimate_action(state=input_state,
                                                                                                                                          is_inference=True)
                nll_val_loss = updater.multivariate_gaussian_nll_loss(y_true=output_action,
                                                                      y_pred=action_mu_and_std)
                cummulative_val_loss += nll_val_loss.item()
        
        # calculate average validation loss in the current epoch
        avg_bc_val_loss_value = round(cummulative_val_loss / len(val_loader), 5)

        print(f"Epoch {epoch + 1}/{constants.BC_NUMBER_EPOCHS}, Batch Train Loss: {avg_bc_train_loss_value}, Batch Validation Loss: {avg_bc_val_loss_value}")
        
        # check for early stopping
        if avg_bc_val_loss_value < best_bc_val_loss:
            best_bc_val_loss = avg_bc_val_loss_value

            # save action policy network parameters after every epoch
            save_policy(epoch=epoch,
                        policy_network=policy_network,
                        saving_path=policy_saving_path,
                        loss_value_str=str(avg_bc_train_loss_value).replace(".", "_"))
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= configs.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss!")
            break

    print("\n================== Training Finished (choo choo) ==================\n")
