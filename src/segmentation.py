# ============================================================
# SEGMENTATION INFERENCE MODULE
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import (
    SEGMENTATION_MODEL_PATH,
    SEGMENTATION_IMAGE_SIZE,
    NUM_SEGMENTATION_CLASSES,
)


# ============================================================
# 1. Simple U-Net model
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
# 2. Load segmentation model
# ============================================================

def load_segmentation_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SEGMENTATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation model not found at {SEGMENTATION_MODEL_PATH}. "
            "Run train_segmentation.py first."
        )

    model = SimpleUNet(num_classes=NUM_SEGMENTATION_CLASSES)

    model.load_state_dict(
        torch.load(
            SEGMENTATION_MODEL_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# 3. Predict segmentation mask
# ============================================================

def predict_segmentation_mask(image_path, model, device):

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    transform = transforms.Compose([
        transforms.Resize(SEGMENTATION_IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    predicted_mask_resized = cv2.resize(
        predicted_mask.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    original_image_np = np.array(image)

    return original_image_np, predicted_mask_resized


# ============================================================
# 4. Mask visualization
# ============================================================

def colorize_segmentation_mask(mask):

    color_map = {
        0: (0, 0, 0),         # background / other = black
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


# ============================================================
# 5. Extract sky crop from segmentation mask
# ============================================================

def extract_sky_crop(original_image, mask):

    sky_mask = mask == 5
    sky_ratio = float(np.sum(sky_mask) / mask.size) if mask.size > 0 else 0.0

    if not np.any(sky_mask):
        return None, sky_ratio

    ys, xs = np.where(sky_mask)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    sky_crop = original_image[y_min:y_max + 1, x_min:x_max + 1]

    if sky_crop.size == 0:
        return None, sky_ratio

    return sky_crop, sky_ratio

# ============================================================
# 5B. Fallback sky crop
# ============================================================

def extract_sky_crop_with_fallback(
    original_image,
    mask,
    min_sky_ratio=0.02,
    fallback_top_ratio=0.35,
):

    sky_crop, sky_ratio = extract_sky_crop(original_image, mask)

    if sky_crop is not None and sky_ratio >= min_sky_ratio:
        return sky_crop, sky_ratio, "segmentation"

    #Fallback: use top part of the original image
    height, width = original_image.shape[:2]
    fallback_height = int(height * fallback_top_ratio)

    fallback_crop = original_image[:fallback_height, :]

    return fallback_crop, sky_ratio, "fallback_top_crop"

# ============================================================
# 6. Save predicted mask
# ============================================================

def save_predicted_mask(mask, output_path):

    color_mask = colorize_segmentation_mask(mask)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), color_mask_bgr)
