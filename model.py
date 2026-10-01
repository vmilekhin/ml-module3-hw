"""Модель: ResNet18 с заменённым fc."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Создаёт ResNet18 с новым fc-слоем на num_classes классов."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
