import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiagonalGaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.log_std = nn.Parameter(torch.full((1, out_dim), -0.5))

    def forward(self, x):
        mean = torch.tanh(self.fc_mean(x))
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6)
        return torch.distributions.Normal(mean, std)

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, alpha):
        super(ActorNetwork, self).__init__()
        channels, height, width = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        with torch.no_grad():
            conv_out = np.prod(self.conv(torch.zeros(1, *input_shape)).size()).item()
            
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            DiagonalGaussian(512, n_actions)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = alpha)
        self.to(device)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
    
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, alpha):
        super(CriticNetwork, self).__init__()
        channels, height, width = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        with torch.no_grad():
            conv_out = np.prod(self.conv(torch.zeros(1, *input_shape)).size()).item()
            
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = alpha)
        self.to(device)  
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x