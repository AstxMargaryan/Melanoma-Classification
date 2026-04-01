import torch.nn as nn
from torchvision import models


# class BaselineCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

#         self.batch_norm_1 = nn.BatchNorm2d(32)
#         self.batch_norm_2 = nn.BatchNorm2d(64)

#         self.max_pool = nn.MaxPool2d(2)
#         self.relu = nn.ReLU()

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()

#         self.linear_1 = nn.Linear(64, 128)
#         self.dropout = nn.Dropout(0.3)
#         self.linear_2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_norm_1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)

#         x = self.conv2(x)
#         x = self.batch_norm_2(x)
#         x = self.relu(x)
#         x = self.max_pool(x)

#         x = self.pool(x)

#         x = self.flatten(x)
#         x = self.linear_1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.linear_2(x)

#         return x



# import torch.nn as nn
# from torchvision import models

# def get_resnet18_model(num_classes=1):
#     model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)

#     return model

# import torch.nn as nn
# from torchvision import models


# def get_resnet50_model(num_classes=1):
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     return model

import timm

import timm
import torch.nn as nn

def get_model(model_name):
    if model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=1)

    elif model_name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
