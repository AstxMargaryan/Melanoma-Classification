# import torch
# import torch.nn as nn


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SimpleCNN, self).__init__()

#         # Convolutional layers
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (B,3,224,224) -> (B,16,224,224)
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # -> (B,16,112,112)

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (B,32,112,112)
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # -> (B,32,56,56)

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (B,64,56,56)
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)   # -> (B,64,28,28)
#         )

#         # Fully connected layers
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 28 * 28, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
    

# import torch
# import torch.nn as nn


# # class SimpleCNN(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
# #         self.relu1 = nn.ReLU()
# #         self.pool1 = nn.MaxPool2d(2)

# #         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
# #         self.relu2 = nn.ReLU()
# #         self.pool2 = nn.MaxPool2d(2)

# #         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# #         self.relu3 = nn.ReLU()
# #         self.pool3 = nn.MaxPool2d(2)

# #         self.flatten = nn.Flatten()
# #         self.fc1 = nn.Linear(64 * 28 * 28, 128)
# #         self.relu4 = nn.ReLU()
# #         self.dropout = nn.Dropout(0.3)
# #         self.fc2 = nn.Linear(128, 2)

# #     def forward(self, x):
# #         x = self.pool1(self.relu1(self.conv1(x)))
# #         x = self.pool2(self.relu2(self.conv2(x)))
# #         x = self.pool3(self.relu3(self.conv3(x)))

# #         x = self.flatten(x)
# #         x = self.relu4(self.fc1(x))
# #         x = self.dropout(x)
# #         x = self.fc2(x)

# #         return x


# import torch.nn as nn


# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 28 * 28, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# import torch.nn as nn


# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),   # 128 -> 64

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),   # 64 -> 32

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 32 -> 16
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# import torch.nn as nn
# from torchvision import models


# def build_model(num_classes=2, dropout=0.3):
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#     in_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(dropout),
#         nn.Linear(in_features, num_classes)
#     )

#     return model



import torch.nn as nn
from torchvision import models

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # փոխում ենք վերջին layer-ը
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    return model