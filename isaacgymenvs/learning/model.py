from skrl.models.torch import GaussianModel
from skrl.models.torch import DeterministicModel
from skrl.utils.model_instantiators import deterministic_model, Shape
import torch.nn as nn
import torch
from gym.spaces import Box
class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.conv = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=24):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 4, 5, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),    
            nn.Conv2d(4, 8, 5, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv(x)

class StochasticActor(GaussianModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", network_features=[512,256,128], activation_function="elu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
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
            self.network.append(Layer(in_channels, feature, activation_function))
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
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))


    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x

class StochasticActorHeightmap(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=1080, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
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

class StochasticActorHeightmapWithMemory(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, num_memories=5, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_memories = num_memories
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception * self.num_memories
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap
        self.cnn = nn.ModuleList()

        # Create encoder for heightmap
        in_channels = int((self.num_exteroception)) #- 2) / 2)
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        # Create CNN
        self.cnn = DoubleConv()

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        #print(states.shape)
        x = states[:,self.num_proprioception:]
        #print(x.shape)
        x = x.reshape(x.shape[0],self.num_exteroception,self.num_memories)
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        #print(x.shape)
        x = x.squeeze()
        #print(x.shape)
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat((states[:,0:self.num_proprioception], x), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter


class StochasticActorHeightmapWithCNN(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=45, num_rows=24, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_rows = num_rows
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception * self.num_rows
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap
        self.cnn = nn.ModuleList()

        # Create encoder for heightmap
        in_channels = 192#276#int((self.num_exteroception * self.num_rows)/4) #- 2) / 2)
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        # Create CNN
        self.cnn = DoubleConv()

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        #print(states.shape)
        x = states[:,self.num_proprioception:]
        #print(x.shape)
        x = x.reshape(x.shape[0],self.num_exteroception,self.num_rows)
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        #print(x.shape)
        #x = x.squeeze()
        x = x.flatten(2,3)
        x = x.flatten(1,2)
        #print(x.shape)
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat((states[:,0:self.num_proprioception], x), dim=1)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter


class StochasticActorHeightmapGLU(GaussianModel):
    def __init__(self, observation_space, action_space, num_exteroception=1080, device = "cuda:0", network_features=[512,256,128], encoder_features=[80,60], activation_function="relu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
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
    def __init__(self, observation_space, action_space, num_exteroception=1080, device = "cuda:0", network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
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

class DeterministicHeightmapWithMemory(DeterministicModel):
    def __init__(self, observation_space, action_space, num_exteroception=150, device = "cuda:0", num_memories=5, network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_memories = num_memories
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception  * self.num_memories
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = int((self.num_exteroception)) #- 2) / 2)
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        # Create CNN
            self.cnn = DoubleConv()
        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        x = x.reshape(x.shape[0],self.num_exteroception,self.num_memories)
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        x = x.squeeze()
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat([states[:,0:self.num_proprioception], x], dim=1)
        for layer in self.network:
            x = layer(x)
        return x

class DeterministicHeightmapWithCNN(DeterministicModel):
    def __init__(self, observation_space, action_space, num_exteroception=45, num_rows=24, device = "cuda:0", num_memories=5, network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_rows = num_rows
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception  * self.num_rows
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = 192#276#int((self.num_exteroception * self.num_rows)/4) #- 2) / 2)
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        # Create CNN
            self.cnn = DoubleConv()
        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = states[:,self.num_proprioception:]
        x = x.reshape(x.shape[0],self.num_exteroception,self.num_rows)
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        #x = x.squeeze()
        x = x.flatten(2,3)
        x = x.flatten(1,2)
        for layer in self.encoder:
            x = layer(x)
        x = torch.cat([states[:,0:self.num_proprioception], x], dim=1)
        for layer in self.network:
            x = layer(x)
        return x
        
class DeterministicHeightmapTD3(DeterministicModel):
    def __init__(self, observation_space, action_space, num_exteroception=1080, device = "cuda:0", network_features=[128,64], encoder_features=[80,60], activation_function="relu", clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)
        self.num_exteroception = num_exteroception  # External information (Heightmap)
        self.num_proprioception = observation_space.shape[0] - self.num_exteroception 
        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap

        # Create encoder for heightmap
        in_channels = self.num_exteroception
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
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
            self.network.append(Layer(in_channels, feature, activation_function))
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
