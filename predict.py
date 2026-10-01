"""Инференс: определение породы по фото."""
import sys
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from model import build_model


def predict_image(image_path: str, model_path: str = "model.pth") -> tuple[str, float]:
    """Возвращает (название породы, уверенность)."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = ResNet18_Weights.DEFAULT.transforms()
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs[0, idx].item()

    return classes[idx], confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py путь/к/фото.jpg")
        sys.exit(1)
    name, conf = predict_image(sys.argv[1])
    print(f"Порода: {name}, уверенность: {conf:.2%}")
