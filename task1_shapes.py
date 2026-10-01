"""Задание 1: shape Tensor после transform и после unsqueeze(0)."""
from PIL import Image
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Создаём тестовую картинку 500x500 (RGB, серый цвет)
image = Image.new("RGB", (500, 500), color=(128, 128, 128))

# После transform
x = transform(image)
print(f"После transform:        {x.shape}")
print(f"  Тип: {type(x)}")
print(f"  dtype: {x.dtype}")

# После unsqueeze(0) — добавляем batch-размер
x_batch = x.unsqueeze(0)
print(f"После unsqueeze(0):     {x_batch.shape}")
