# ============================================================
# TRAIN WEATHER MODEL
# Trains a CNN classifier using the folders in data/weather/
#
# Expected folder structure:
# data/weather/
# ├── cloudy/
# ├── rain/
# ├── shine/
# └── sunrise/
# ============================================================

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import (
    WEATHER_DATA_DIR,
    WEATHER_MODEL_PATH,
    WEATHER_CLASSES_PATH,
    WEATHER_IMAGE_SIZE,
    WEATHER_BATCH_SIZE,
    WEATHER_EPOCHS,
    WEATHER_LEARNING_RATE,
    WEATHER_VALIDATION_SPLIT,
    RANDOM_STATE,
    OUTPUTS_DIR,
)


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Weather CNN model
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
# 3. Weather dataset
# ============================================================

class WeatherImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(self.labels[idx])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 4. Load image paths and labels from folders
# ============================================================

def load_weather_files(weather_data_dir: Path):
    image_paths = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_folders = sorted(
        [folder for folder in weather_data_dir.iterdir() if folder.is_dir()]
    )

    if not class_folders:
        raise FileNotFoundError(
            f"No class folders found in {weather_data_dir}. "
            "Expected folders like cloudy/, rain/, shine/, sunrise/."
        )

    for class_folder in class_folders:
        class_name = class_folder.name

        for image_path in class_folder.rglob("*"):
            if image_path.suffix.lower() in valid_extensions:
                image_paths.append(image_path)
                labels.append(class_name)

    if not image_paths:
        raise FileNotFoundError(
            f"No weather images found inside {weather_data_dir}."
        )

    return image_paths, labels


# ============================================================
# 5. Training function
# ============================================================

def train_weather_model():
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("\nLoading weather dataset from:")
    print(WEATHER_DATA_DIR)

    image_paths, text_labels = load_weather_files(WEATHER_DATA_DIR)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(text_labels)
    weather_classes = label_encoder.classes_

    print("\nWeather classes:")
    for idx, class_name in enumerate(weather_classes):
        print(f"{idx}: {class_name}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        encoded_labels,
        test_size=WEATHER_VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=encoded_labels,
    )

    weather_transform = transforms.Compose([
        transforms.Resize(WEATHER_IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = WeatherImageDataset(
        train_paths,
        train_labels,
        transform=weather_transform,
    )

    val_dataset = WeatherImageDataset(
        val_paths,
        val_labels,
        transform=weather_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=WEATHER_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=WEATHER_BATCH_SIZE,
        shuffle=False,
    )

    print("\nTraining samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    num_classes = len(weather_classes)

    model = WeatherCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=WEATHER_LEARNING_RATE,
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\nStarting weather model training...\n")

    for epoch in range(WEATHER_EPOCHS):
        # -----------------------------
        # Training phase
        # -----------------------------
        model.train()
        total_train_loss = 0.0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{WEATHER_EPOCHS} [Train]",
        )

        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -----------------------------
        # Validation phase
        # -----------------------------
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{WEATHER_EPOCHS} [Val]",
            ):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{WEATHER_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # ========================================================
    # Save model and classes
    # ========================================================

    WEATHER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), WEATHER_MODEL_PATH)
    np.save(WEATHER_CLASSES_PATH, weather_classes)

    print("\nSaved weather model to:")
    print(WEATHER_MODEL_PATH)

    print("\nSaved weather classes to:")
    print(WEATHER_CLASSES_PATH)

    # ========================================================
    # Save training plots
    # ========================================================

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    loss_plot_path = OUTPUTS_DIR / "weather_training_loss.png"
    accuracy_plot_path = OUTPUTS_DIR / "weather_validation_accuracy.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Weather Model Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Weather Model Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_plot_path, bbox_inches="tight")
    plt.close()

    print("\nSaved weather training plots to:")
    print(loss_plot_path)
    print(accuracy_plot_path)

    print("\nWeather training finished successfully.")


# ============================================================
# 6. Run script
# ============================================================

if __name__ == "__main__":
    train_weather_model()