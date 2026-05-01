# ============================================================
# VISUALIZATION MODULE
# Creates final visual demo panels for road danger predictions.
# ============================================================

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation import colorize_segmentation_mask


# ============================================================
# 1. Overlay segmentation mask
# ============================================================

def create_segmentation_overlay(original_image, mask, alpha=0.45):
    """
    Creates an RGB image with the segmentation mask overlaid
    on top of the original image.
    """

    color_mask = colorize_segmentation_mask(mask)

    overlay = cv2.addWeighted(
        original_image.astype(np.uint8),
        1 - alpha,
        color_mask.astype(np.uint8),
        alpha,
        0,
    )

    return overlay


# ============================================================
# 2. Create final demo panel
# ============================================================

def create_demo_panel(
    original_image,
    mask,
    yolo_image,
    result,
    output_path,
):
    """
    Saves a 2x2 panel:
    - original image
    - segmentation overlay
    - YOLO detections
    - final score text
    """

    segmentation_overlay = create_segmentation_overlay(original_image, mask)

    final_score = result.get("final_danger_score", 0.0)
    danger_level = result.get("danger_level", "Unknown")

    visual_score = result.get("visual_risk_score", 0.0)
    weather_score = result.get("weather_score", 0.0)
    audio_score = result.get("audio_score", 0.0)

    predicted_weather = result.get("predicted_weather", "Not used")
    weather_confidence = result.get("weather_confidence", 0.0)

    num_people = result.get("num_people", 0)
    num_vehicles = result.get("num_vehicles", 0)
    num_bikes = result.get("num_bikes", 0)

    text_lines = [
        f"Final danger score: {final_score:.3f}",
        f"Danger level: {danger_level}",
        "",
        f"Visual risk: {visual_score:.3f}",
        f"Weather risk: {weather_score:.3f}",
        f"Audio risk: {audio_score:.3f}",
        "",
        f"Predicted weather: {predicted_weather}",
        f"Weather confidence: {weather_confidence:.3f}",
        "",
        f"People detected: {num_people}",
        f"Vehicles detected: {num_vehicles}",
        f"Bikes detected: {num_bikes}",
    ]

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(segmentation_overlay)
    ax2.set_title("Segmentation Overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(yolo_image)
    ax3.set_title("YOLO Detections")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    ax4.set_title("Danger Prediction Summary")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ============================================================
# 3. Save sky crop if available
# ============================================================

def save_sky_crop(sky_crop, output_path):
    """
    Saves extracted sky crop if it exists.
    """

    if sky_crop is None:
        return False

    sky_crop_bgr = cv2.cvtColor(sky_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), sky_crop_bgr)

    return True