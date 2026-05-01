# ============================================================
# CONFIGURATION FILE
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Main project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

CITYSCAPES_DIR = DATA_DIR / "cityscapes"

CITYSCAPES_IMAGE_ROOT = CITYSCAPES_DIR / "Cityscape Dataset" / "leftImg8bit"
CITYSCAPES_MASK_ROOT = CITYSCAPES_DIR / "Fine Annotations" / "gtFine"

TRAIN_IMAGES_DIR = CITYSCAPES_IMAGE_ROOT / "train"
TRAIN_MASKS_DIR = CITYSCAPES_MASK_ROOT / "train"

VAL_IMAGES_DIR = CITYSCAPES_IMAGE_ROOT / "val"
VAL_MASKS_DIR = CITYSCAPES_MASK_ROOT / "val"

WEATHER_DATA_DIR = DATA_DIR / "weather"

INPUT_IMAGES_DIR = DATA_DIR / "input_images"
INPUT_AUDIO_DIR = DATA_DIR / "input_audio"

# ------------------------------------------------------------
# Model output paths
# ------------------------------------------------------------

SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_model_improved.pth"
WEATHER_MODEL_PATH = MODELS_DIR / "weather_model.pth"
WEATHER_CLASSES_PATH = MODELS_DIR / "weather_classes.npy"

YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# ------------------------------------------------------------
# Output folders
# ------------------------------------------------------------

PREDICTED_MASKS_DIR = OUTPUTS_DIR / "predicted_masks"
DEMO_IMAGES_DIR = OUTPUTS_DIR / "demo_images"
RESULTS_CSV_PATH = OUTPUTS_DIR / "results.csv"

# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

SEGMENTATION_IMAGE_SIZE = (256, 256)
WEATHER_IMAGE_SIZE = (128, 128)

NUM_SEGMENTATION_CLASSES = 6

# Grouped segmentation classes:
# 0 = background / other
# 1 = drivable area
# 2 = sidewalk
# 3 = vulnerable road users
# 4 = vehicles
# 5 = sky

SEGMENTATION_CLASS_NAMES = {
    0: "background_other",
    1: "drivable_area",
    2: "sidewalk",
    3: "vulnerable_road_users",
    4: "vehicles",
    5: "sky",
}

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

SEGMENTATION_BATCH_SIZE = 4
SEGMENTATION_EPOCHS = 5
SEGMENTATION_LEARNING_RATE = 1e-3
SEGMENTATION_MAX_SAMPLES = 300

WEATHER_BATCH_SIZE = 16
WEATHER_EPOCHS = 10
WEATHER_LEARNING_RATE = 1e-3
WEATHER_VALIDATION_SPLIT = 0.2

RANDOM_STATE = 42

# ------------------------------------------------------------
# Pipeline settings
# ------------------------------------------------------------

MIN_SKY_RATIO_FOR_WEATHER = 0.02

# ------------------------------------------------------------
# Create folders automatically
# ------------------------------------------------------------

for folder in [
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTED_MASKS_DIR,
    DEMO_IMAGES_DIR,
    INPUT_IMAGES_DIR,
    INPUT_AUDIO_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)