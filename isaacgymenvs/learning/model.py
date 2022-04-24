from skrl.models.torch import GaussianModel
from skrl.models.torch import DeterministicModel
from skrl.utils.model_instantiators import deterministic_model, Shape
import torch.nn as nn
import torch
from gym.spaces import Box
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Conv,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh()
           } 
        self.conv = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.conv(x)

class StochasticActor(GaussianModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", network_features=[512,256,128], activation_function="elu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in network_features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))


    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter
        

class StochasticCritic(DeterministicModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x

class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))


    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x

# class DeterministicCritic(DeterministicModel):
#     def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False):
#         super().__init__(observation_space, action_space, device, clip_actions)

#         self.network = nn.ModuleList()
        

#         in_channels = observation_space.shape[0]
#         for feature in features:
#             self.network.append(Conv(in_channels, feature, activation_function))
#             in_channels = feature

#         self.network.append(nn.Linear(in_channels,1))


#     def compute(self, states, taken_actions):
#         x = states
#         for layer in self.network:
#             x = layer(x)
#         return x

class StochasticActorHeightmap(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Conv(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat((states[:,0:self.num_proprioception], x), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter


class StochasticActorHeightmapGLU(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Conv(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]*2))
        self.network.append(nn.GLU())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat((states[:,0:self.num_proprioception], x), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter



class DeterministicHeightmap(DeterministicModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, device = "cuda:0", network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Conv(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat([states[:,0:self.num_proprioception], x], dim=1)
        for layer in self.network:
            x = layer(x)
        return x

class DeterministicHeightmapTD3(DeterministicModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, device = "cuda:0", network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Conv(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat([states[:,0:self.num_proprioception,taken_actions], x], dim=1)
        for layer in self.network:
            x = layer(x)
        return x

class DeterministicCritic(DeterministicModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]+self.num_actions
        for feature in features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = torch.cat([states, taken_actions], dim=1)
        for layer in self.network:
            x = layer(x)
        return x
# #REMOVE LATER
# class DeterministicActor(DeterministicModel):
#     def __init__(self, observation_space, action_space, device, clip_actions = False):
#         super().__init__(observation_space, action_space, device, clip_actions)
#         in_channels = observation_space.shape[0]
#         self.linear_layer_1 = nn.Linear(in_channels, 32)
#         self.linear_layer_2 = nn.Linear(32, 32)
#         self.action_layer = nn.Linear(32, self.num_actions)

#     def compute(self, states, taken_actions):
#         x = nn.functional.elu(self.linear_layer_1(states))
#         x = nn.functional.elu(self.linear_layer_2(x))
#         return torch.tanh(self.action_layer(x))

# #REMOVE LATER
# class Critic(DeterministicModel):
#     def __init__(self, observation_space, action_space, device, clip_actions = False):
#         super().__init__(observation_space, action_space, device, clip_actions)
#         in_channels = observation_space.shape[0]
#         self.net = nn.Sequential(nn.Linear(in_channels + self.num_actions, 32),
#                                  nn.ELU(),
#                                  nn.Linear(32, 32),
#                                  nn.ELU(),
#                                  nn.Linear(32, 1))

#     def compute(self, states, taken_actions):
#         return self.net(torch.cat([states, taken_actions], dim=1))        

if __name__ == "__main__":
    pass
    # observation_space = Box(-torch.inf,torch.inf,(3,))
    # action_space = Box(-1.0,1.0,(2,))
    # policy = Policy(observation_space,action_space, features=[512,256,128], activation_function="elu")
    # value = Value(observation_space,action_space, features=[512,256,128], activation_function="elu")
    # policy2 = deterministic_model(observation_space=env.observation_space, 
    #                             action_space=env.action_space,
    #                             device=device,
    #                             clip_actions=False, 
    #                             input_shape=Shape.OBSERVATIONS,
    #                             hiddens=[64, 64],
    #                             hidden_activation=["relu", "relu"],
    #                             output_shape=Shape.ACTIONS,
    #                             output_activation=None,
    #                             output_scale=1.0)


    # print(policy)
    # print(value)
