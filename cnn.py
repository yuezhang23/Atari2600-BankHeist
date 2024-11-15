import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, in_channels=3, num_actions=18):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        def conv2d_size_out(self, size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        self.fc = nn.Linear(3136, num_actions)
        
        convd = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convd * convd * 64
        self.fc = nn.Linear(linear_input_size, num_actions) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0), -1))
        return x
