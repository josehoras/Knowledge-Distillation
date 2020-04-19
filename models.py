import torch
import torch.nn as nn

class linear_net(nn.Module):
    def __init__(self, dropout=0.5):
        super(linear_net, self).__init__()
        self.linear_1 = nn.Linear(784, 1200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(1200, 1200)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_3 = nn.Linear(1200, 10)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        scores = self.relu(scores)
        scores = self.dropout(scores)
        scores = self.linear_3(scores)
        return scores


class small_linear_net(nn.Module):
    def __init__(self):
        super(small_linear_net, self).__init__()
        self.linear_1 = nn.Linear(784, 50)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(50, 10)
        # self.dropout = nn.Dropout(p=0.5)
        # self.dropout = nn.Dropout(p=0.5)
        # self.linear_3 = nn.Linear(50, 10)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        # scores = self.relu(scores)
        # scores = self.dropout(scores)
        # scores = self.linear_3(scores)
        return scores


#Convolution layers
class convolutional_net(nn.Module):
    def __init__(self):
        super(convolutional_net, self).__init__()
        self.convolution = nn.Conv2d(1, 10, 5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(5)
        self.linear = nn.Linear(160, 10)

    def forward(self, input):
        in_reshaped = input.reshape(-1, 1, 28, 28)
        x_conv = self.convolution(in_reshaped)
        x_relu = self.relu(x_conv)
        x_conv_out = self.maxpool(x_relu)
        x_flat = x_conv_out.reshape(-1, 160)
        scores = self.linear(x_flat)
        return scores