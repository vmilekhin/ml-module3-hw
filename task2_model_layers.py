"""Задание 2: найти conv1, layer1..layer4, avgpool, fc в ResNet18."""
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Печатаем всю модель (будет длинно)
print("=" * 60)
print("Полная модель ResNet18:")
print("=" * 60)
print(model)

print("\n" + "=" * 60)
print("Ключевые слои:")
print("=" * 60)
print(f"conv1:   {model.conv1}")
print(f"layer1:  {model.layer1}")
print(f"layer2:  {model.layer2}")
print(f"layer3:  {model.layer3}")
print(f"layer4:  {model.layer4}")
print(f"avgpool: {model.avgpool}")
print(f"fc:      {model.fc}")
