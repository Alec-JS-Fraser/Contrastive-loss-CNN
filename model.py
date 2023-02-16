import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        return x


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_cnn = BaseCNN()
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x1, x2):
        output1 = self.base_cnn(x1)
        output2 = self.base_cnn(x2)
        distance = torch.abs(output1 - output2)
        similarity = self.fc2(distance)
        return similarity


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, label):
        loss = torch.mean(0.5
                          * ((1-label) * torch.pow(similarity, 2)  # positive loss
                             + label * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)) # negative loss
                          )
        return loss

# net = SiameseNetwork()
# summary(net, [(3, 224, 224), (3, 224, 224)])