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
            "relu" : nn.ReLU(),
            "elu" : nn.ELU()
           } 
        self.conv = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.conv(x)

class Policy(GaussianModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space.shape[0]))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))


    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter


class Value(DeterministicModel):
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


class DeterministicPolicy(DeterministicModel):
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

class DeterministicValue(DeterministicModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False):
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

class DeterministicValueWithTakenActions(DeterministicModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu",clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.network = nn.ModuleList()
        

        in_channels = observation_space.shape[0]
        for feature in features:
            self.network.append(Conv(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,1))


    def compute(self, states, taken_actions):
        x = torch.cat([states, taken_actions], dim=1)
        for layer in self.network:
            x = layer(x)
        return x

#REMOVE LATER
class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)
        in_channels = observation_space.shape[0]
        self.linear_layer_1 = nn.Linear(in_channels, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.action_layer = nn.Linear(32, self.num_actions)

    def compute(self, states, taken_actions):
        x = nn.functional.elu(self.linear_layer_1(states))
        x = nn.functional.elu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x))

#REMOVE LATER
class Critic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)
        in_channels = observation_space.shape[0]
        self.net = nn.Sequential(nn.Linear(in_channels + self.num_actions, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))        

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
