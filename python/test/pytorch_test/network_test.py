import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution kernel
        # 四个方向都填充了1; 输入数据中 高是1，用2个卷积核输出，卷积核大小2*2; 由于输入2张图片，那么输出格式应该是2* (2*6*6)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, padding=10)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=2)  # 输出应该是2* (16*2*2)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        y1 = self.conv1(x)

        # 卷积是有步长的，能有交集，池化不能有交集
        x = F.max_pool2d(F.relu(y1), (2, 2))
        # print(self.conv1.weight)
        print(y1.size())
        print(x.size())

        # If the size is a square you can only specify a single number
        y2 = self.conv2(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(y2.size())
        print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# stride 步幅
net = Net()
print(net)

# 2张图片，输入通道：高度是1，高度5，宽度5
x = torch.ones(2, 1, 10, 10)
print(x)
x = net.forward(x)
print(x)

params=list(net.parameters())
print(len(params))
# print(params)
for i in params:
    print(i.size())

