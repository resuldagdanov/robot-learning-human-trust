# Contents
* [Installation](#installation)
    - [Anaconda Environment Creation](#anaconda-environment-creation)
    - [Package Installation](#package-installation)
* [General Framework](#general-framework)
* [Environment](#environment)
    - [Download Collected Dataset](#download-collected-dataset)
* [Experiments](#experiments)
    - [Dataset Analysis](#dataset-analysis)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
* [Learning Trust](#learning-trust)
* [Results](#results)
    - [Experiment Outcomes](#experiment-outcomes)
    - [Optimized Model Parameters](#optimized-model-parameters)
* [Corresponding Author](#corresponding-author)

---
# Installation

## Anaconda Environment Creation
> Requires Python >= 3.10:
* `conda create -y -n trust_learning python=3.10`
* `conda activate trust_learning`

## Package Installation
> Execute setup from the source code root of the repository:
* `cd src`
* `pip install -e .`

---
# General Framework
<figure>
    <p align="center">
        <img src="../presentation/images/framework.png" width="1080px" alt="General Framework"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 1: Visualization of the Proposed Framework</figcaption>
</figure>

---
# Environment
> A markov decision process (MDP) environment script is located in the following directory:
* `cd src/environment`
<figure>
    <p align="center">
        <img src="../presentation/images/experiment_environment.JPG" width="640px" alt="Experiment Environment"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 2: Environment of Tiling Operation with Cobot (@ UTS - Robotics Institute)</figcaption>
</figure>

## Download Collected Dataset
> Human collection dataset of demonstrations is shared in the repository:
* `cd dataset/human_demonstrations`

> Training and testing datasets of .json files are located in the following directory:
* `cd dataset/human_demonstrations/2024_01_23_Train/jsons`
* `cd dataset/human_demonstrations/2024_02_02_Test/jsons`

---
# Experiments
<figure>
    <p align="center">
        <img src="../presentation/images/operation_environment.PNG" width="640px" alt="Operation Environment"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 3: Environment of Human-Robot Teaming in Construction (@ UTS - Robotics Institute)</figcaption>
</figure>
<figure>
    <p align="center">
        <img src="../presentation/images/data_collection.PNG" width="640px" alt="Data Collection"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 4: Human Demonstration Data Collection Process</figcaption>
</figure>

## Dataset Analysis
> Priorly collected dataset is shared in the repository:
* `cd dataset/human_demonstrations`
<figure>
    <p align="center">
        <img src="../presentation/images/visualize_training_dataset.PNG" width="640px" alt="Training Dataset"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 5: Human Collected Training Dataset of 43 Trajectories (Expert Demonstrations)</figcaption>
</figure>

> The dataset is analyzed in the following notebook:
* `cd src/analyses`
* `jupyter notebook`
* `visualize_demonstration.ipynb`
<figure>
    <p align="center">
        <img src="../presentation/images/visualize_particular_trajectory.png" width="640px" alt="Particular Trajectory"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 6: Visualization of One Trajectory of Human Operation (Expert Demonstration)</figcaption>
</figure>

## Model Training
> Neural network model files are located in the following directory:
* `cd src/models`

> To run the training script:
* `cd src/optimization`
* `python train.py`

> To run the training of the policy model (explicitly):
* `cd src/optimization`
* `python train_policy.py`

> To run the training of the reward model (explicitly):
* `cd src/optimization`
* `python train_reward.py`

## Model Evaluation
> The evaluation of the trained models is included in the following notebook:
* `cd src/evaluation`
* `jupyter notebook`
* `evaluate_trust_estimation.ipynb`
<figure>
    <p align="center">
        <img src="../presentation/images/optimization_result.png" width="640px" alt="Optimization Result"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 7: Parameter Optimization during Robot Learning of Human Trust after Each Experiment</figcaption>
</figure>

<figure>
    <p align="center">
        <img src="../presentation/images/measurement_and_reward.png" width="640px" alt="Measurement and Reward"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 8: Relationship Between Human Trust Measurements and Learned Reward Function</figcaption>
</figure>
<figure>
    <p align="center">
        <img src="../presentation/images/learning_trust_distribution.png" width="640px" alt="Trust Distributions"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 9: Shifts in Trust Distribution During Human Trust Learning Stage Experiments</figcaption>
</figure>

> A Beta Reputation System implementation script is located in the following directory:
* `cd src/trusts`
* `model_dynamics.py`

---
# Learning Trust
<figure>
    <p align="center">
        <img src="../presentation/images/methodology.jpg" width="640px" alt="Methodology Framework"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 10: Illustration of an Iterative Trust Estimation Process</figcaption>
</figure>

---
# Results
> Visualize the results of the Modeled Human Trust in the following notebook:
* `cd src/evaluation`
* `jupyter notebook`
* `visualize_trust_dynamics.ipynb`
<figure>
    <p align="center">
        <img src="../presentation/images/inference_experiment.png" width="640px" alt="Inference Experiment"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 11: Comparing Trust Estimation with Trust Measurement in Verification Stage Experiments</figcaption>
</figure>
<figure>
    <p align="center">
        <img src="../presentation/images/inference_result.png" width="1080px" alt="Inference Result"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 12: Reward Function and Probabilistic Trust Estimation in the Verification Stage Experiments</figcaption>
</figure>

## Experiment Outcomes
> The resultant Excel files of the experiments are shared in the following directory:
* `cd results/experiments`

> Trust learning stage experiment results:
* `cd results/experiments/learning_experiments`
* `results/experiments/learning_stage_experiment_results.xlsx`

> Inference stage experiment results:
* `cd results/experiments/inference_experiments`
* `results/experiments/inference_stage_experiment_results.xlsx`

## Optimized Model Parameters
> The trained policy network and reward model parameters are shared in the following directories:
* `cd results/policy_network_params`
* `cd results/reward_network_params`

---
# Corresponding Author
> For any inquiries or lack of clarity, please contact the corresponding author:
➔ [Resul.Dagdanov@uts.edu.au](mailto:Resul.Dagdanov@uts.edu.au)