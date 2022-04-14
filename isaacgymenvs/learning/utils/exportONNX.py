import isaacgym

import torch
from utils.model import GaussianPolicyHeightmap, GaussianPolicy, GaussianPolicyHeightmap
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
env = load_isaacgym_env_preview3(task_name="Exomy_actual")

env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
model = GaussianPolicy(env.observation_space, env.action_space, features=[256,160,128], activation_function="relu")
model = GaussianPolicyHeightmap(env.observation_space, env.action_space, network_features=[256,160,128], activation_function="relu")
print(model)

model.eval()
dummy_input = torch.randn(1, 73)
#dummy_input = torch.tensor([[512.0, 512.0, 512.0]])
input_names = [ "actual_input" ]
output_names = [ "output" ]




torch.onnx.export(model, 
                  dummy_input,
                  "ModelWithEncoder.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )