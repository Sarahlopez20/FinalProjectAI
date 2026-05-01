# ============================================================
# COLAB TRAIN SEGMENTATION MODEL
# Trains a small U-Net on Cityscapes to predict grouped masks.
#
# Grouped classes:
# 0 = background / other
# 1 = drivable area
# 2 = sidewalk
# 3 = vulnerable road users
# 4 = vehicles
# 5 = sky
# ============================================================


# ============================================================
# 0. Mount Google Drive
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


# ============================================================
# 1. Imports
# ============================================================

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# 2. GPU check
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Go to Runtime > Change runtime type > T4 GPU.")


# ============================================================
# 3. Paths
# Corrected for your Google Drive folder structure
# ============================================================

PROJECT_DIR = Path("/content/drive/MyDrive/AI Project")

CITYSCAPES_DIR = PROJECT_DIR / "City scape"

TRAIN_IMAGES_DIR = CITYSCAPES_DIR / "Cityscape Dataset" / "leftImg8bit" / "train"
VAL_IMAGES_DIR = CITYSCAPES_DIR / "Cityscape Dataset" / "leftImg8bit" / "val"

TRAIN_MASKS_DIR = CITYSCAPES_DIR / "Fine Annotations" / "gtFine" / "train"
VAL_MASKS_DIR = CITYSCAPES_DIR / "Fine Annotations" / "gtFine" / "val"

OUTPUTS_DIR = PROJECT_DIR / "outputs_colab"
MODELS_DIR = PROJECT_DIR / "models_colab"

SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_model_improved.pth"

SEGMENTATION_IMAGE_SIZE = (256, 256)
NUM_SEGMENTATION_CLASSES = 6
SEGMENTATION_BATCH_SIZE = 4
SEGMENTATION_EPOCHS = 25
SEGMENTATION_LEARNING_RATE = 1e-3
SEGMENTATION_MAX_SAMPLES = 1000
RANDOM_STATE = 42

print("\nChecking paths:")
print("PROJECT_DIR:", PROJECT_DIR, PROJECT_DIR.exists())
print("CITYSCAPES_DIR:", CITYSCAPES_DIR, CITYSCAPES_DIR.exists())
print("TRAIN_IMAGES_DIR:", TRAIN_IMAGES_DIR, TRAIN_IMAGES_DIR.exists())
print("TRAIN_MASKS_DIR:", TRAIN_MASKS_DIR, TRAIN_MASKS_DIR.exists())
print("VAL_IMAGES_DIR:", VAL_IMAGES_DIR, VAL_IMAGES_DIR.exists())
print("VAL_MASKS_DIR:", VAL_MASKS_DIR, VAL_MASKS_DIR.exists())


# ============================================================
# 4. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. Cityscapes label mapping
# ============================================================

GROUPED_CLASS_MAP = {
    7: 1,    # road -> drivable area
    8: 2,    # sidewalk -> sidewalk

    24: 3,   # person -> vulnerable road user
    25: 3,   # rider -> vulnerable road user
    32: 3,   # motorcycle -> vulnerable road user
    33: 3,   # bicycle -> vulnerable road user

    26: 4,   # car -> vehicle
    27: 4,   # truck -> vehicle
    28: 4,   # bus -> vehicle
    31: 4,   # train -> vehicle

    23: 5,   # sky -> sky
}


def remap_cityscapes_mask(mask_np):
    """
    Converts original Cityscapes label IDs into grouped project classes.
    Any label not in GROUPED_CLASS_MAP becomes 0 = background / other.
    """

    remapped = np.zeros_like(mask_np, dtype=np.uint8)

    for original_id, grouped_id in GROUPED_CLASS_MAP.items():
        remapped[mask_np == original_id] = grouped_id

    return remapped


# ============================================================
# 6. Collect image and mask paths
# ============================================================

def collect_cityscapes_pairs(images_dir: Path, masks_dir: Path, max_samples=None):
    image_paths = sorted(images_dir.rglob("*_leftImg8bit.png"))

    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    pairs = []

    for image_path in image_paths:
        city_name = image_path.parent.name

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png",
        )

        mask_path = masks_dir / city_name / mask_name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            "No valid Cityscapes image/mask pairs found. "
            "Check your Cityscapes folder paths."
        )

    return pairs


# ============================================================
# 7. Dataset with augmentation
# ============================================================

class CityscapesGroupedDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

        self.image_resize = transforms.Resize(SEGMENTATION_IMAGE_SIZE)
        self.mask_resize = transforms.Resize(
            SEGMENTATION_IMAGE_SIZE,
            interpolation=Image.NEAREST,
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def apply_augmentation(self, image, mask):
        """
        Applies the same geometric augmentation to image and mask.
        Color augmentation only affects the image.
        """

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)

            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        image = self.to_tensor(image)

        mask_np = np.array(mask)
        grouped_mask = remap_cityscapes_mask(mask_np)
        grouped_mask = torch.tensor(grouped_mask, dtype=torch.long)

        return image, grouped_mask


# ============================================================
# 8. Simple U-Net model
# IMPORTANT: must match your local segmentation.py architecture
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        bottleneck = self.bottleneck(p3)

        u3 = self.up3(bottleneck)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.final_conv(u1)


# ============================================================
# 9. Metrics and losses
# ============================================================

def pixel_accuracy(outputs, masks):
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == masks).sum().item()
    total = masks.numel()

    return correct / total if total > 0 else 0.0


def mean_iou(outputs, masks, num_classes):
    predictions = torch.argmax(outputs, dim=1)

    ious = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        true_class = masks == class_id

        intersection = (pred_class & true_class).sum().item()
        union = (pred_class | true_class).sum().item()

        if union > 0:
            ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return float(np.mean(ious))


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(
            targets,
            num_classes=self.num_classes,
        )

        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


def compute_class_weights(num_classes):
    weights = torch.ones(num_classes, dtype=torch.float32)

    weights[0] = 0.4   # background / other
    weights[1] = 1.0   # road
    weights[2] = 1.2   # sidewalk
    weights[3] = 3.0   # vulnerable road users
    weights[4] = 1.8   # vehicles
    weights[5] = 1.0   # sky

    return weights


# ============================================================
# 10. Colorize mask for visual checking
# ============================================================

def colorize_segmentation_mask(mask):
    color_map = {
        0: (0, 0, 0),          # background / other = black
        1: (128, 64, 128),    # road = purple
        2: (244, 35, 232),    # sidewalk = pink
        3: (220, 20, 60),     # vulnerable users = red
        4: (0, 0, 142),       # vehicles = blue
        5: (70, 130, 180),    # sky = light blue
    }

    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color

    return color_mask


def save_validation_example(model, val_dataset, device, output_path):
    model.eval()

    image_tensor, true_mask = val_dataset[0]

    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    image_np = image_tensor.permute(1, 2, 0).numpy()
    true_mask_np = true_mask.numpy()

    pred_color = colorize_segmentation_mask(pred_mask)
    true_color = colorize_segmentation_mask(true_mask_np)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_color)
    plt.title("True grouped mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_color)
    plt.title("Predicted grouped mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# 11. Training function
# ============================================================

def train_segmentation_model():
    set_seed(RANDOM_STATE)

    print("\nCollecting Cityscapes training pairs...")
    train_pairs = collect_cityscapes_pairs(
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        max_samples=SEGMENTATION_MAX_SAMPLES,
    )

    print("Collecting Cityscapes validation pairs...")
    val_pairs = collect_cityscapes_pairs(
        VAL_IMAGES_DIR,
        VAL_MASKS_DIR,
        max_samples=100,
    )

    print("Training pairs:", len(train_pairs))
    print("Validation pairs:", len(val_pairs))

    train_dataset = CityscapesGroupedDataset(train_pairs, augment=True)
    val_dataset = CityscapesGroupedDataset(val_pairs, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=SEGMENTATION_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=SEGMENTATION_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = SimpleUNet(num_classes=NUM_SEGMENTATION_CLASSES).to(device)

    class_weights = compute_class_weights(NUM_SEGMENTATION_CLASSES).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes=NUM_SEGMENTATION_CLASSES)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SEGMENTATION_LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_ious = []

    best_val_loss = float("inf")
    best_val_iou = 0.0

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting segmentation training...\n")

    for epoch in range(SEGMENTATION_EPOCHS):
        # -----------------------------
        # Training phase
        # -----------------------------
        model.train()
        total_train_loss = 0.0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{SEGMENTATION_EPOCHS} [Train]",
        )

        for images, masks in train_progress:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss_ce = ce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)

            loss = 0.7 * loss_ce + 0.3 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_progress.set_postfix(
                loss=round(loss.item(), 4),
                ce=round(loss_ce.item(), 4),
                dice=round(loss_dice.item(), 4),
            )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -----------------------------
        # Validation phase
        # -----------------------------
        model.eval()
        total_val_loss = 0.0
        total_accuracy = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{SEGMENTATION_EPOCHS} [Val]",
            ):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss_ce = ce_loss(outputs, masks)
                loss_dice = dice_loss(outputs, masks)
                loss = 0.7 * loss_ce + 0.3 * loss_dice

                total_val_loss += loss.item()
                total_accuracy += pixel_accuracy(outputs, masks)
                total_iou += mean_iou(
                    outputs,
                    masks,
                    NUM_SEGMENTATION_CLASSES,
                )

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_accuracy / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        val_ious.append(avg_val_iou)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{SEGMENTATION_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Pixel Accuracy: {avg_val_accuracy:.4f} | "
            f"Val Mean IoU: {avg_val_iou:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_iou = avg_val_iou

            torch.save(model.state_dict(), SEGMENTATION_MODEL_PATH)

            print("Saved new best segmentation model:")
            print(SEGMENTATION_MODEL_PATH)
            print("Best Val Loss:", round(best_val_loss, 4))
            print("Best Val Mean IoU:", round(best_val_iou, 4))

    # ========================================================
    # Save plots
    # ========================================================

    loss_plot_path = OUTPUTS_DIR / "segmentation_training_loss.png"
    accuracy_plot_path = OUTPUTS_DIR / "segmentation_validation_accuracy.png"
    iou_plot_path = OUTPUTS_DIR / "segmentation_validation_iou.png"
    example_plot_path = OUTPUTS_DIR / "segmentation_validation_example.png"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Segmentation Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label="Validation pixel accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel accuracy")
    plt.title("Segmentation Validation Pixel Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_plot_path, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_ious, label="Validation mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Segmentation Validation Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(iou_plot_path, bbox_inches="tight")
    plt.show()
    plt.close()

    # Load best model again before creating visual example
    model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=device))
    save_validation_example(model, val_dataset, device, example_plot_path)

    print("\nSaved model to:")
    print(SEGMENTATION_MODEL_PATH)

    print("\nSaved plots to:")
    print(loss_plot_path)
    print(accuracy_plot_path)
    print(iou_plot_path)
    print(example_plot_path)

    print("\nBest validation loss:", round(best_val_loss, 4))
    print("Best validation mean IoU:", round(best_val_iou, 4))

    print("\nSegmentation training finished successfully.")


# ============================================================
# 12. Run training
# ============================================================

train_segmentation_model()