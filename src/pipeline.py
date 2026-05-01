# ============================================================
# FULL ROAD DANGER PREDICTION PIPELINE
# ============================================================

from pathlib import Path
import cv2
import pandas as pd

from config import (
    PREDICTED_MASKS_DIR,
    DEMO_IMAGES_DIR,
    MIN_SKY_RATIO_FOR_WEATHER,
)

from src.segmentation import (
    load_segmentation_model,
    predict_segmentation_mask,
    extract_sky_crop_with_fallback,
    save_predicted_mask,
)

from src.weather import (
    load_weather_model,
    predict_weather_from_image,
)

from src.yolo_detection import (
    load_yolo_model,
    run_yolo_detection,
    draw_yolo_detections,
)

from src.audio import analyze_audio_optional

from src.scoring import (
    compute_segmentation_features,
    compute_visual_risk,
    compute_weather_risk,
    compute_audio_risk,
    compute_final_danger_score,
)

from src.visualization import (
    create_demo_panel,
    save_sky_crop,
)

# ============================================================
# GLOBAL RISK HISTORY (for temporal smoothing)
# ============================================================

risk_history = []


# ============================================================
# 1. Load all models once
# ============================================================

def load_all_models():
    print("Loading segmentation model...")
    segmentation_model, segmentation_device = load_segmentation_model()

    print("Loading weather model...")
    weather_model, weather_classes, weather_device = load_weather_model()

    print("Loading YOLO model...")
    yolo_model = load_yolo_model()

    print("All models loaded successfully.")

    return {
        "segmentation_model": segmentation_model,
        "segmentation_device": segmentation_device,
        "weather_model": weather_model,
        "weather_classes": weather_classes,
        "weather_device": weather_device,
        "yolo_model": yolo_model,
    }


# ============================================================
# 2. Predict one image
# ============================================================

def predict_dangerousness(image_path, models, audio_path=None):
    global risk_history

    image_path = Path(image_path)
    print(f"\nProcessing image: {image_path.name}")

    # ---------------- SEGMENTATION ----------------
    original_image, predicted_mask = predict_segmentation_mask(
        image_path=image_path,
        model=models["segmentation_model"],
        device=models["segmentation_device"],
    )

    segmentation_features = compute_segmentation_features(predicted_mask)

    mask_output_path = PREDICTED_MASKS_DIR / f"{image_path.stem}_mask.png"
    save_predicted_mask(predicted_mask, mask_output_path)

    # ---------------- YOLO ----------------
    yolo_result = run_yolo_detection(
        image_path=image_path,
        model=models["yolo_model"],
    )

    detections = yolo_result["detections"]
    yolo_features = yolo_result["yolo_features"]

    image_bgr = cv2.imread(str(image_path))
    yolo_image = draw_yolo_detections(image_bgr, detections)

    # ---------------- WEATHER ----------------
    sky_crop, sky_ratio, sky_crop_source = extract_sky_crop_with_fallback(
        original_image=original_image,
        mask=predicted_mask,
        min_sky_ratio=MIN_SKY_RATIO_FOR_WEATHER,
        fallback_top_ratio=0.35,
    )

    weather_prediction = {
        "predicted_weather": None,
        "weather_confidence": 0.0,
        "weather_probabilities": {},
    }

    weather_used = False

    if sky_crop is not None:
        weather_prediction = predict_weather_from_image(
            image=sky_crop,
            model=models["weather_model"],
            weather_classes=models["weather_classes"],
            device=models["weather_device"],
        )

        weather_used = True

        sky_crop_path = DEMO_IMAGES_DIR / f"{image_path.stem}_sky_crop.png"
        save_sky_crop(sky_crop, sky_crop_path)

    weather_risk = compute_weather_risk(
        predicted_weather=weather_prediction["predicted_weather"],
        confidence=weather_prediction["weather_confidence"],
    )

    weather_risk["weather_used"] = weather_used

    # ---------------- AUDIO ----------------
    audio_result = analyze_audio_optional(audio_path)
    audio_risk = compute_audio_risk(audio_result)
    print(f"Audio score for {image_path.name}: {audio_risk['audio_score']}")

    # ---------------- VISUAL RISK ----------------
    visual_risk = compute_visual_risk(
        segmentation_features=segmentation_features,
        yolo_features=yolo_features,
    )

    # ---------------- FINAL SCORE ----------------
    final_score = compute_final_danger_score(
        visual_risk_score=visual_risk["visual_risk_score"],
        weather_score=weather_risk["weather_score"],
        audio_score=audio_risk["audio_score"],
        weather_used=weather_risk["weather_used"],
        audio_used=audio_risk["audio_used"],
    )

    # ---------------- TEMPORAL SMOOTHING ----------------
    risk_history.append(final_score["final_danger_score"])

    #We only keep only last 4 values
    if len(risk_history) > 4:
        risk_history.pop(0)

    if len(risk_history) == 4:
        mean_risk_score = sum(risk_history) / 4
    else:
        mean_risk_score = None

    if mean_risk_score is not None:
        final_score["mean_risk_score"] = float(mean_risk_score)
    else:
        final_score["mean_risk_score"] = None
    # ---------------- RESULT ----------------
    result = {
        "image_path": str(image_path),
        "image_name": image_path.name,

        **segmentation_features,
        **yolo_features,

        "num_detections": len(detections),

        **visual_risk,

        "sky_ratio": sky_ratio,
        "sky_crop_source": sky_crop_source,
        **weather_prediction,
        **weather_risk,

        **audio_risk,

        **final_score,

        "mean_risk_score": final_score["mean_risk_score"],

        "predicted_mask_path": str(mask_output_path),
    }

    # ---------------- SAVE DEMO ----------------
    demo_output_path = DEMO_IMAGES_DIR / f"{image_path.stem}_demo.png"

    create_demo_panel(
        original_image=original_image,
        mask=predicted_mask,
        yolo_image=yolo_image,
        result=result,
        output_path=demo_output_path,
    )

    result["demo_image_path"] = str(demo_output_path)

    return result


# ============================================================
# 3. Predict multiple images
# ============================================================

def predict_folder(image_paths, audio_path=None):
    models = load_all_models()

    results = []

    for image_path in image_paths:
        result = predict_dangerousness(
            image_path=image_path,
            models=models,
            audio_path=audio_path,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df
