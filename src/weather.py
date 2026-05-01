# ============================================================
# WEATHER INFERENCE MODULE
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import (
    WEATHER_MODEL_PATH,
    WEATHER_CLASSES_PATH,
    WEATHER_IMAGE_SIZE,
)


# ============================================================
# 1. Weather CNN model
# Must match the architecture used in train_weather.py
# ============================================================

class WeatherCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flattened_size = 64 * (WEATHER_IMAGE_SIZE[0] // 8) * (WEATHER_IMAGE_SIZE[1] // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. Load weather model
# ============================================================

def load_weather_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not WEATHER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Weather model not found at {WEATHER_MODEL_PATH}. "
            "Run train_weather.py first."
        )

    if not WEATHER_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Weather classes file not found at {WEATHER_CLASSES_PATH}. "
            "Run train_weather.py first."
        )

    weather_classes = np.load(WEATHER_CLASSES_PATH, allow_pickle=True)
    model = WeatherCNN(num_classes=len(weather_classes))

    model.load_state_dict(
        torch.load(
            WEATHER_MODEL_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, weather_classes, device


# ============================================================
# 3. Predict weather from an image
# ============================================================

def predict_weather_from_image(image, model, weather_classes, device):
    """
    image can be:
    - PIL image
    - numpy array
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(WEATHER_IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities).item())
    predicted_class = str(weather_classes[predicted_idx])
    confidence = float(probabilities[predicted_idx].item())

    all_probabilities = {
        str(weather_classes[i]): float(probabilities[i].item())
        for i in range(len(weather_classes))
    }

    return {
        "predicted_weather": predicted_class,
        "weather_confidence": confidence,
        "weather_probabilities": all_probabilities,
    }