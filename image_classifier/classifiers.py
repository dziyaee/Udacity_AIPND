# Pytorch imports
import torch
import torch.nn.functional as F
from torch import nn


# Squeezenet Classifier
class SN_Classifier(nn.Module):

    def __init__(self, net, hp):
        super().__init__()

        # In Features, Out Features, and Tuple of Hidden Units
        in_features = net['in_features']
        out_features = net['out_features']
        hidden_units = hp['hidden_units']

        # Define Linear Layer params
        self.fc1 = nn.Linear(in_features, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], out_features)

        # Weight Mean, Weight STD, and Bias Value init params
        init_params = hp['init_params']

        # Initialize params if given values
        if init_params is not None:

            # mean, std, bias values for param initialization
            mean, std, bias = init_params

            # Param initialization
            for name, child in self.named_children():

                # Init weights with normal distribution using given mean and std values
                torch.nn.init.normal_(child.weight, mean, std)

                # Init bias with constant value using given bias value
                torch.nn.init.constant_(child.bias, bias)

        # Dropout probability
        p_dropout = hp['p_dropout']

        # Define dropout using given dropout probability p_dropout
        self.dropout = nn.Dropout(p_dropout)

        # Avgpool to change shape of conv layer output from (B x 512 x 13 x 13) to (B x 512)
        self.avgpool2d = nn.AvgPool2d(13)

    def forward(self, x):

        # Changes conv layer output shape from (B x 512 x 13 x 13 ) to (B x 512 x 1 x 1)
        x = self.avgpool2d(x)

        # Squeezes output from (B x 512 x 1 x 1) to (B x 512 x 1). dim=2 is specified to avoid squeezing B in case of B = 1
        x = torch.squeeze(x, dim=2)

        # Squeezes output from (B x 512 x 1) to (B x 512). dim=2 is specified to avoid squeezing B in case of B = 1
        x = torch.squeeze(x, dim=2)

        # First 2 Layers: Activation Function = ReLU. Dropout = Active
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Final Layer: LogSoftMax. dim=1 is specified to calculate log probabilities per data sample across columns
        x = F.log_softmax(self.fc3(x), dim = 1)


        return x


# Densenet Classifier
class DN_Classifier(nn.Module):

    def __init__(self, net, hp):
        super().__init__()

        # In Features, Out Features, and Tuple of Hidden Units
        in_features = net['in_features']
        out_features = net['out_features']
        hidden_units = hp['hidden_units']

        # Define Linear Layer params
        self.fc1 = nn.Linear(in_features, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], out_features)

        # Weight Mean, Weight STD, and Bias Value init params
        init_params = hp['init_params']

        # Initialize params if given values
        if init_params is not None:

            # mean, std, bias values for param initialization
            mean, std, bias = init_params

            # Param initialization
            for name, child in self.named_children():

                # Init weights with normal distribution using given mean and std values
                torch.nn.init.normal_(child.weight, mean, std)

                # Init bias with constant value using given bias value
                torch.nn.init.constant_(child.bias, bias)

        # Dropout probability
        p_dropout = hp['p_dropout']

        # Define dropout using given dropout probability p_dropout
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):

        # First 2 Layers: Activation Function = ReLU. Dropout = Active
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Final Layer: LogSoftMax. dim=1 is specified to calculate log probabilities per data sample across columns
        x = F.log_softmax(self.fc3(x), dim = 1)

        return x
