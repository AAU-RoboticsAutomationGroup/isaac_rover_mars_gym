
import hydra
import isaacgym
from omegaconf import DictConfig, OmegaConf
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3
from skrl.memories.torch import RandomMemory
from utils.model import StochasticActor, StochasticCritic, DeterministicActor, DeterministicCritic, StochasticActorHeightmap
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer

env = load_isaacgym_env_preview3(task_name="Exomy_actual")

env = wrap_env(env)

device = env.device

# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
#hydra.compose
#@hydra.main(config_path="./cfg", config_name="config")
def run_tests():
    #config = OmegaConf.to_yaml(cfg)
    path = 'cfg/tests/test.yaml'
    cfg = OmegaConf.load(path)
    timesteps = 100000
    agent = []
    for test in cfg['tests']:
        # Load config for current test
        config = cfg['tests'][test]
        # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
        memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
        # Get model for reinforcement learning algorithm
        model = get_model(config)
        # Get config file for reinforcement learning algo
        model_cfg = get_cfg(config)
        # Configure how often to save
        model_cfg["experiment"]["write_interval"] = 120
        model_cfg["experiment"]["checkpoint_interval"] = 3000
        model_cfg["experiment"]["experiment_name"] = config['algorithm'] +'Actor' + str(config['actor_mlp']) + config['activation_function'] + '_Critic' + str(config['critic_mlp']) + config['activation_function_critic'] + '_Encoder' + str(config['encoder_mlp']) + config['activation_function_encoder'] + '_' + test + '_step' + str(timesteps)
        agent = get_agent(config, model, memory, model_cfg, env)
        cfg_trainer = {"timesteps": timesteps, "headlesAs": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        trainer.train()


def get_model(config):
    if config['algorithm'] == 'ppo':
        # PPO requires 2 models, visit its documentation for more details
        # https://skrl.readthedocs.io/en/laconfig/modules/skrl.agents.ppo.html#spaces-and-models
        models = {  "policy": StochasticActorHeightmap(env.observation_space, env.action_space, network_features=config['actor_mlp'], encoder_features=config['encoder_mlp'], activation_function=config['activation_function']),
                    "value": StochasticCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic'])}

        # Initialize the models' parameters (weights and biases) using a Gaussian distribution
        for model in models.values():
            print("hej")
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)   
        print('ppo')
    elif config['algorithm'] == 'TD3':
        models = {  "policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
                        "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
                        "critic_1": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "critic_2": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic'])}
        
        # Initialize the models' parameters (weights and biases) using a Gaussian distribution
        for model in models.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)  
    elif config['algorithm'] == 'DDPG':
        models = { "policy": DeterministicActor(env.observation_space, env.action_space, features=config['actor_mlp'], activation_function=config['activation_function']),
                        "target_policy": DeterministicActor(env.observation_space, env.action_space, features=config['actor_mlp'], activation_function=config['activation_function']),
                        "critic": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "target_critic": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic'])}
        
        # Initialize the models' parameters (weights and biases) using a Gaussian distribution
        for model in models.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)  
    elif config['algorithm'] == 'SAC':
        models = {  "policy": StochasticActor(env.observation_space, env.action_space, features=config['actor_mlp'], activation_function=config['activation_function']),
                        "critic_1": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "critic_2": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic']),
                        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, features=config['critic_mlp'], activation_function=config['activation_function_critic'])}
        
        # Initialize the models' parameters (weights and biases) using a Gaussian distribution
        for model in models.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)  
    return models

def get_cfg(config):
        if config['algorithm'] == 'ppo':
            cfg_ppo = PPO_DEFAULT_CONFIG.copy()
            cfg_ppo["rollouts"] = 16
            cfg_ppo["learning_epochs"] = 4
            cfg_ppo["mini_batches"] = 2
            cfg_ppo["discount_factor"] = 0.99
            cfg_ppo["lambda"] = 0.99
            cfg_ppo["policy_learning_rate"] = 0.0003
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
            return cfg_ppo
        elif config['algorithm'] == 'TD3':
            cfg_td3 = TD3_DEFAULT_CONFIG.copy()
            cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.2, device=device)
            cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
            cfg_td3["smooth_regularization_clip"] = 0.1
            cfg_td3["gradient_steps"] = 1
            cfg_td3["batch_size"] = 512
            cfg_td3["random_timesteps"] = 0
            cfg_td3["learning_starts"] = 0
            return cfg_td3  
        elif config['algorithm'] == 'DDPG':
            cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
            cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
            cfg_ddpg["gradient_steps"] = 1
            cfg_ddpg["batch_size"] = 512
            cfg_ddpg["random_timesteps"] = 0
            cfg_ddpg["learning_starts"] = 0
            return cfg_ddpg

        elif config['algorithm'] == 'SAC':
            cfg_sac = SAC_DEFAULT_CONFIG.copy()
            cfg_sac["gradient_steps"] = 1
            cfg_sac["batch_size"] = 512
            cfg_sac["random_timesteps"] = 0
            cfg_sac["learning_starts"] = 0
            cfg_sac["learn_entropy"] = True
            return cfg_sac

def get_agent(config, models, memory, cfg, env):
    if config['algorithm'] == 'ppo':
        agent_ppo = PPO(models=models, 
                        memory=memory, 
                        cfg=cfg, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space,
                        device=env.device)    
        return agent_ppo

    elif config['algorithm'] == 'TD3':
        agent_td3 = TD3(models=models, 
                        memory=memory, 
                        cfg=cfg, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space,
                        device=env.device) 
        return agent_td3  

    elif config['algorithm'] == 'DDPG':
        agent_DDPG = DDPG(models=models, 
                memory=memory, 
                cfg=cfg, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=env.device) 
        return agent_DDPG

    elif config['algorithm'] == 'SAC':
        agent_SAC = SAC(models=models, 
                        memory=memory, 
                        cfg=cfg, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space,
                        device=env.device) 
        return agent_SAC

if __name__ == "__main__":
    run_tests()