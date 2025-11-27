import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torch.hub import load_state_dict_from_url

class BirdEmbedder(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()
        
        # 1. Backbone (EfficientNet B0)
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Manual Weight Loading (Docker Safe)
        if pretrained:
            try:
                state = load_state_dict_from_url(weights.url, check_hash=False)
                self.backbone.load_state_dict(state)
                print("✅ ImageNet weights loaded.")
            except:
                print("⚠️ Failed to load weights, training from scratch.")

        # 2. Remove the old classifier head
        # EfficientNet's last layer is 'classifier'
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove original head

        # 3. The Projection Head (Maps features to 512-D Vector)
        # BN + PReLU helps shape the latent space better than just Linear
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, embedding_size)
            # Note: No Softmax here! We want raw vectors.
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        # Normalize to the Hypersphere (Length = 1)
        # This is crucial for ArcFace/Cosine Similarity
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
