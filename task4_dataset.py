"""Задание 4: свой Dataset-класс с __len__ и __getitem__."""
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights


class SyntheticDataset(Dataset):
    """Простой Dataset: генерирует картинки по индексу."""

    def __init__(self, size: int, transform) -> None:
        self.size = size
        self.transform = transform
        # Заранее подготовим список "классов"
        self.labels = [i % 3 for i in range(size)]  # классы 0, 1, 2

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        # Создаём картинку, зависящую от idx
        color = (
            (idx * 37) % 256,
            (idx * 73) % 256,
            (idx * 113) % 256,
        )
        image = Image.new("RGB", (300, 300), color=color)
        x = self.transform(image)
        y = self.labels[idx]
        return x, y


if __name__ == "__main__":
    transform = ResNet18_Weights.DEFAULT.transforms()
    ds = SyntheticDataset(size=5, transform=transform)

    print(f"Длина датасета: {len(ds)}")
    print(f"ds[0]: x.shape = {ds[0][0].shape}, y = {ds[0][1]}")
    print(f"ds[3]: x.shape = {ds[3][0].shape}, y = {ds[3][1]}")

    # Проверим, что итерация работает
    for i in range(len(ds)):
        x, y = ds[i]
        print(f"  [{i}] x.shape={tuple(x.shape)}, y={y}")
