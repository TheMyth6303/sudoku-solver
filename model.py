import torch.nn as nn
import torch.nn.functional as F


conv1_k = 3
conv1_c = 16
pool1_k = 2
conv2_k = 5
conv2_c = 32
pool2_k = 2
conv3_k = 5
conv3_c = 32
fc1_out = 24
input = 50
output = 9


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv1_c, kernel_size=conv1_k)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_k, stride=pool1_k)
        self.conv2 = nn.Conv2d(in_channels=conv1_c,
                               out_channels=conv2_c, kernel_size=conv2_k)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_k, stride=pool2_k)
        self.conv3 = nn.Conv2d(in_channels=conv2_c,
                               out_channels=conv3_c, kernel_size=conv3_k)
        self.fc1 = nn.Linear(in_features=int(
            (((input-conv1_k+1)/pool1_k-conv2_k+1)/pool2_k-conv3_k+1)**2)*conv2_c, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc1_out, out_features=output)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x))
        y = self.pool1(y)
        y = F.leaky_relu(self.conv2(y))
        y = self.pool2(y)
        y = self.conv3(y)
        y = y.view(-1, int((((input-conv1_k+1)/pool1_k-conv2_k+1) /
                   pool2_k-conv3_k+1)**2)*conv2_c)
        y = F.leaky_relu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return y
