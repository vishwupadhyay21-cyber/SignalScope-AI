"""Transfer-learning neural network classifier architecture for SignalScope."""

import torch
import torch.nn as nn
from torchvision import models
from model.config import CLASSIFIER_DROPOUT


class SignalScopeClassifier(nn.Module):
    """ResNet-18 Transfer Learning Classifier for Real vs AI-Generated Image Forensics."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = CLASSIFIER_DROPOUT,
    ) -> None:
        """Initializes ResNet-18 backbone with custom high-capacity classifier head."""
        super().__init__()

        # Load ResNet-18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        # Replace classification head with deep dense classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(256, 1)  # Output raw logit for single binary class (1 = AI-generated, 0 = Real)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw scalar logits."""
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the target convolutional feature layer for Grad-CAM explainability."""
        return self.backbone.layer4[-1]



def build_model(pretrained: bool = True) -> SignalScopeClassifier:
    """Factory function to build and return a SignalScopeClassifier model instance."""
    return SignalScopeClassifier(pretrained=pretrained)
