import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.pool(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


def get_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model