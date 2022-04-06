from skrl.models.torch import GaussianModel
from skrl.models.torch import DeterministicModel
import torch.nn as nn


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, activation_function="elu"):
        super(Conv,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh()
        }

        self.conv = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
            #nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Policy(GaussianModel):
    def __init__(self, observation_space, action_space, device = "cuda:0", features=[512,256,128], activation_function="elu", clip_actions=False, clip_log_std = True, min_log_std= -20.0, max_log_std = 2.0) -> None:
        super().__init__(observation_space, action_space, device, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.network = nn.ModuleList()

        in_channels = observation_space
        for feature in features:
            self.network.append(Conv(in_channels, feature,activation_function=activation_function))
            in_channels = feature
        
        self.final_conv = self.network.append(nn.Linear(in_channels,action_space))

    def compute(self, states, taken_actions):
        x = states
        for layer in self.network:
            x = layer(x)
        return x


class Value(DeterministicModel):
    def __init__(self, observation_space, action_space , device = "cuda:0", clip_actions= False):
        super().__init__(observation_space, action_space, device, clip_actions)
        

if __name__ == "__main__":
    model = Policy(2,3,features=[256,128], activation_function="relu")
    print(model)