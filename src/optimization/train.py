import os
import sys

import torch

import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from updater import Updater

from torch.utils.tensorboard import SummaryWriter

# to add the parent "models" folder to sys path and import neural networks
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.policy_model import RobotPolicy
from models.reward_model import RewardFunction

from utils.config import Config


def load_dataset(configs):
    return None


if __name__ == "__main__":
    print("\n================== Initializing Training (chuff chuff) ! ==================\n")

    configs = Config()
    # call the parameters method to set the parameters
    configs.parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: ", device)
    configs.device = device

    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 device=configs.device)
    reward_network = RewardFunction(state_action_size=configs.state_size + configs.action_size,
                                    hidden_size=configs.hidden_size,
                                    out_size=configs.reward_size,
                                    device=configs.device)

    updater = Updater(policy_model=policy_network,
                      reward_model=reward_network,
                      configs=configs)

    demonstration_data = load_dataset(configs=configs)
    dataset_loader = DataLoader(demonstration_data,
                                batch_size=configs.batch_size,
                                shuffle=True,
                                num_workers=0)

    for batch_data in tqdm(dataset_loader):

        demo_state = batch_data["demo_state"].float().to(device)
        demo_action = batch_data["demo_action"].float().to(device)

        robot_action = policy_network(demo_state)