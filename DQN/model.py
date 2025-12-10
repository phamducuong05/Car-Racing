import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        channels, height, width = input_shape

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv_output_size(conv_output_size(conv_output_size(height, 8, 4), 4, 2), 3, 1)
        conv_w = conv_output_size(conv_output_size(conv_output_size(width, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_h * conv_w * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        q_values = self.fc_out(x)
        return q_values