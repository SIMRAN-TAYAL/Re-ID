import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(ReIDModel, self).__init__()

        # Backbone: ResNet-50
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the classifier layer (fc)
        self.base = nn.Sequential(*list(base_model.children())[:-2])

        # Adaptive Pooling to get fixed-size features
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Embedding head (reduces 2048 -> 512)
        self.embedding = nn.Linear(2048, embedding_dim)

        # Batch normalization
        self.bn = nn.BatchNorm1d(embedding_dim)

        # Classification head (for ID classification loss)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x, return_feature=False):
        x = self.base(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        feat = self.embedding(x)
        feat = self.bn(feat)

        if return_feature:
            return feat  # used during inference (for embedding extraction)

        cls_score = self.classifier(feat)
        return cls_score, feat
