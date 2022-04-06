import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3
from utils.model import DeterministicActor, Critic
from gym.spaces import Box
from skrl.utils.model_instantiators import deterministic_model, Shape

# Load and wrap the Isaac Gym environment.
# The following lines are intended to support both versions (preview 2 and 3). 
# It tries to load from preview 3, but if it fails, it will try to load from preview 2

env = load_isaacgym_env_preview3(task_name="Exomy_actual")

env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_td3 = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
              "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
              "critic_1": Critic(env.observation_space, env.action_space, device),
              "critic_2": Critic(env.observation_space, env.action_space, device),
              "target_critic_1": Critic(env.observation_space, env.action_space, device),
              "target_critic_2": Critic(env.observation_space, env.action_space, device)}

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_td3.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)   


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.2, device=device)
cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
cfg_td3["smooth_regularization_clip"] = 0.1
cfg_td3["gradient_steps"] = 1
cfg_td3["batch_size"] = 512
cfg_td3["random_timesteps"] = 0
cfg_td3["learning_starts"] = 0
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_td3["experiment"]["write_interval"] = 25
cfg_td3["experiment"]["checkpoint_interval"] = 1000


agent = TD3(models=models_td3,
            memory=memory, 
            cfg=cfg_td3, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()