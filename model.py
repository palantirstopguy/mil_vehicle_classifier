import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_model(num_classes):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model