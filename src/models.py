import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes, use_attention=False, num_layers=3):
        super().__init__()
        self.use_attention = use_attention

        layers =[nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)]
        in_channels = 32

        for _ in range(1, num_layers):
            out_channels = in_channels * 2
            layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.attention = ChannelAttention(in_channels) if use_attention else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        return self.classifier(x)

class EmbeddingNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, embedding_size)
        )
    def forward(self, x):
        emb = self.head(self.backbone(x))
        return torch.nn.functional.normalize(emb, p=2, dim=1)

def get_model(model_name, num_classes, use_attention=False, num_layers=3):
    if model_name == "custom":
        return CustomCNN(num_classes, use_attention, num_layers)
    elif model_name == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == "embedding":
        return EmbeddingNet(num_classes)
    raise ValueError(f"Model {model_name} unbekannt.")
