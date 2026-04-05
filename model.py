import timm
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.batch_norm_3 = nn.BatchNorm2d(128)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.4)
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

        x = self.conv3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.max_pool(x) 

        x = self.pool(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


def get_model(model_name: str):

    if model_name == "cnn":
        return BaselineCNN()

    elif model_name == "resnet50":
        model = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=1
        )

    elif model_name == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=1
        )


    elif model_name == "efficientnet_b3":
            model = timm.create_model(
                "efficientnet_b3",
                pretrained=True,
                num_classes=1,
                drop_rate=0.4,      
                drop_path_rate=0.2  
            )

    elif model_name == "efficientnet_b4":
        model = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=1
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

    