import numpy as np
import pandas as pd
import torch

from typing import List
from torch.utils.data import Dataset
from utils import common, constants


class PolicyDatasetLoader(Dataset):

    def __init__(self,
                 demo_data_json_paths: List[str]):
        
        self.state_columns = constants.STATE_COLUMNS
        self.action_columns = constants.ACTION_COLUMNS

        self.state_norms = [constants.MAX_DISTANCE_TO_OBJECT,
                            constants.MAX_DISTANCE_TO_TARGET,
                            constants.MAX_DISTANCE_TO_START,
                            constants.MAX_DISTANCE_TO_GROUND]
        
        self.action_norms = [constants.END_EFFECTOR_POSITION_RANGE_X,
                             constants.END_EFFECTOR_POSITION_RANGE_Y,
                             constants.END_EFFECTOR_POSITION_RANGE_Z]
        
        self.trajectory_length = constants.TRAJECTORY_SIZE
        
        self.demo_state_data, self.demo_action_data = self.load_data(json_paths=demo_data_json_paths,
                                                                     column_names=constants.COLUMN_NAMES)
        
        self.dataset_size = len(self.demo_state_data) if not self.demo_state_data.empty else 0

        print("\n================== Policy Dataset Loader ==================\n")
        print("Number of Trajectories: ", len(demo_data_json_paths))
        print("Each Trajectory Length: ", self.trajectory_length)
        print("Full Demo Dataset Size: ", self.dataset_size)
    
    def __len__(self):

        return self.dataset_size
    
    def __getitem__(self,
                    idx: int):
        
        # get normalized state and action vectors as dataframe
        sample_states = self.demo_state_data.iloc[idx].astype(np.float64)
        sample_actions = self.demo_action_data.iloc[idx].astype(np.float64)
        
        # convert state and action to torch tensors
        state_tensor = torch.FloatTensor(sample_states.values.flatten())
        action_tensor = torch.FloatTensor(sample_actions.values.flatten())

        return state_tensor, action_tensor
    
    def load_data(self,
                  json_paths: List[str],
                  column_names: List[str]) -> (pd.DataFrame, pd.DataFrame):
        
        state_dfs, action_dfs = [], []

        # loop through every trajectory data specified in json_paths
        for json_path in json_paths:
            df = common.json2dataframe(json_path=json_path,
                                       column_names=column_names)
            
            # discritize the trajectory into equally spaced points
            df = common.discritize_dataframe(df=df,
                                             return_n_rows=self.trajectory_length)
            
            state_dfs.append(
                common.extract_state_vector(df=df,
                                            state_columns=self.state_columns,
                                            norm_value_list=self.state_norms))
            action_dfs.append(
                common.extract_action_vector(df=df,
                                             action_columns=self.action_columns,
                                             norm_range_list=self.action_norms))
        
        # all trajectories are concatenated into one dataframe
        concatenated_state_df = pd.concat(state_dfs,
                                          ignore_index=True)
        concatenated_action_df = pd.concat(action_dfs,
                                          ignore_index=True)

        return concatenated_state_df, concatenated_action_df
