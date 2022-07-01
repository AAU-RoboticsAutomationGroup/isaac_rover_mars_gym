import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3
from learning.model import StochasticActorHeightmap, DeterministicHeightmap
from gym.spaces import Box
from skrl.utils.model_instantiators import deterministic_model, Shape

# Load and wrap the Isaac Gym environment.
# The following lines are intended to support both versions (preview 2 and 3). 
# It tries to load from preview 3, but if it fails, it will try to load from preview 2
env = load_isaacgym_env_preview3(task_name="Exomy_actual")

env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=30, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=[256,160,128], encoder_features=[60,20], activation_function="elu"),
                "value": None}

# load checkpoint
models_ppo["policy"].load("./runs/3000_policy.pt")
print(models_ppo)



# # Initialize the models' parameters (weights and biases) using a Gaussian distribution
# for model in models_ppo.values():
#     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)   


# Configure and instantiate the agent.z
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
# logging to TensorBoard and write checkpoints each 120 and 3000 timesteps respectively
cfg_ppo["random_timesteps"] = 0
cfg_ppo["experiment"]["write_interval"] = 16
cfg_ppo["experiment"]["checkpoint_interval"] = 0
cfg_ppo["experiment"]["experiment_name"] = "REMOVE"
agent = PPO(models=models_ppo,
            memory=memory, 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headlesAs": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.eval()
print("Done1")
