"""Обучение модели."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet18_Weights

from dataset import FiveBreeds
from model import build_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dog_breeds = ["Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier"]

    train_all = datasets.OxfordIIITPet(
        root="./data", split="trainval", target_types="category", download=True
    )
    name_to_old = {n: i for i, n in enumerate(train_all.classes)}
    selected_old = [name_to_old[n] for n in dog_breeds]
    old_to_new = {old: new for new, old in enumerate(selected_old)}

    transform = ResNet18_Weights.DEFAULT.transforms()
    train_ds = FiveBreeds(train_all, transform, selected_old, old_to_new)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = build_model(num_classes=len(dog_breeds)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch={epoch+1} avg_loss={running / len(train_loader):.4f}")

    torch.save({"model_state": model.state_dict(), "classes": dog_breeds}, "model.pth")
    print("saved: model.pth")


if __name__ == "__main__":
    main()
