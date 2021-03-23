from devices import *
import torch.nn.functional as F
from torch import nn



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution
        self.conv1 = nn.Conv2d(1, 64, 5)  # input channel=1, output channel=64, kernal size = 3*3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 5)
        # fully connected
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # 32 channel, 4 * 4 size(經過Convolution部分後剩4*4大小)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)

    def forward(self, x):
        # state size. 28 * 28(input image size = 28 * 28)
        x = self.pool(F.relu(self.conv1(x)))
        # state size. 12 * 12
        x = self.pool(F.relu(self.conv2(x)))
        # state size. 4 * 4
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
