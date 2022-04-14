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
from learning.model import StochasticActor, StochasticCritic,StochasticActorHeightmap
from gym.spaces import Box
from skrl.utils.model_instantiators import deterministic_model, Shape

# Load and wrap the Isaac Gym environment.
# The following lines are intended to support both versions (preview 2 and 3). 
# It tries to load from preview 3, but if it fails, it will try to load from preview 2
env = load_isaacgym_env_preview3(task_name="Exomy_actual")

env = wrap_env(env)

device = env.device
for i in range(0,3):

    # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


    # Instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
    models_ppo = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=[512,256,128], encoder_features=[80,60], activation_function="relu"),
                    "value": StochasticCritic(env.observation_space, env.action_space, features=[128,64], activation_function="relu")}

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_ppo.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)   


    # Configure and instantiate the agent.
    # Only modify some of the default configuration, visit its documentation to see all the options
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 16
    cfg_ppo["learning_epochs"] = 4
    cfg_ppo["mini_batches"] = 2
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.99
    cfg_ppo["policy_learning_rate"] = 0.003
    cfg_ppo["value_learning_rate"] = 0.0003
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["entropy_loss_scale"] = 0.0
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0.008
    # logging to TensorBoard and write checkpoints each 120 and 3000 timesteps respectively
    cfg_ppo["experiment"]["write_interval"] = 120
    cfg_ppo["experiment"]["checkpoint_interval"] = 3000
    cfg_ppo["experiment"]["experiment_name"] = "tester"
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
    trainer.train()
    print("Done1")
