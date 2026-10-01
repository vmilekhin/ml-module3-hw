"""Dataset для классификации пород собак."""
from torch.utils.data import Dataset


class FiveBreeds(Dataset):
    """Обёртка над базовым датасетом: оставляет только выбранные породы."""

    def __init__(self, base, transform, selected_old, old_to_new) -> None:
        self.base = base
        self.transform = transform
        self.selected_old = selected_old
        self.old_to_new = old_to_new
        self.indices = [
            i for i, (_, y) in enumerate(base) if y in selected_old
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, old_y = self.base[self.indices[idx]]
        image = self.transform(image)
        new_y = self.old_to_new[old_y]
        return image, new_y
