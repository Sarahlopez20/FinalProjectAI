import numpy as np


# ============================================================
# 1. Utility functions
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    """
    Keeps a value within a fixed range.
    Used mainly for final safety.
    """
    return max(minimum, min(maximum, value))


def soft_normalize(value):
    """
    Softly converts a raw risk value into a 0-1 score.

    Unlike hard clipping, this avoids turning every value above 1
    into exactly 1. Risk still increases, but more progressively.

    Example:
    raw = 0.5  -> score = 0.33
    raw = 1.0  -> score = 0.50
    raw = 2.0  -> score = 0.67
    raw = 4.0  -> score = 0.80
    """

    value = max(0.0, value)
    return value / (1.0 + value)


# ============================================================
# 2. Visual risk from segmentation mask
# ============================================================

def compute_segmentation_features(mask):
    """
    Converts the grouped segmentation mask into interpretable ratios.

    Expected mask classes:
    0 = background / other
    1 = drivable area
    2 = sidewalk
    3 = vulnerable road users
    4 = vehicles
    5 = sky
    """

    total_pixels = mask.size

    if total_pixels == 0:
        return {
            "road_ratio": 0.0,
            "sidewalk_ratio": 0.0,
            "vulnerable_ratio": 0.0,
            "vehicle_ratio": 0.0,
            "sky_ratio": 0.0,
        }

    road_ratio = np.sum(mask == 1) / total_pixels
    sidewalk_ratio = np.sum(mask == 2) / total_pixels
    vulnerable_ratio = np.sum(mask == 3) / total_pixels
    vehicle_ratio = np.sum(mask == 4) / total_pixels
    sky_ratio = np.sum(mask == 5) / total_pixels

    return {
        "road_ratio": float(road_ratio),
        "sidewalk_ratio": float(sidewalk_ratio),
        "vulnerable_ratio": float(vulnerable_ratio),
        "vehicle_ratio": float(vehicle_ratio),
        "sky_ratio": float(sky_ratio),
    }


def compute_visual_risk(segmentation_features, yolo_features=None):
    """
    Computes visual risk using segmentation and YOLO features.

    Segmentation provides scene-level information:
    - how much of the image is occupied by vehicles
    - how much by vulnerable road users
    - how urban/pedestrian the scene looks through sidewalk presence

    YOLO provides object-level information:
    - number of people
    - number of vehicles
    - number of bikes
    - size of the largest nearby vehicle

    Vehicles are weighted moderately because they appear in both
    segmentation and YOLO outputs.
    """

    vehicle_ratio = segmentation_features.get("vehicle_ratio", 0.0)
    vulnerable_ratio = segmentation_features.get("vulnerable_ratio", 0.0)
    sidewalk_ratio = segmentation_features.get("sidewalk_ratio", 0.0)

    # Segmentation risk:
    # - vehicle_ratio reduced to avoid double-counting with YOLO vehicles
    # - vulnerable users remain highly weighted
    # - sidewalk is a weak urban-context signal
    segmentation_risk = (
        1.8 * vehicle_ratio
        + 3.0 * vulnerable_ratio
        + 0.4 * sidewalk_ratio
    )

    yolo_risk = 0.0

    if yolo_features is not None:
        num_people = yolo_features.get("num_people", 0)
        num_vehicles = yolo_features.get("num_vehicles", 0)
        num_bikes = yolo_features.get("num_bikes", 0)
        largest_vehicle_area_ratio = yolo_features.get("largest_vehicle_area_ratio", 0.0)

        # YOLO risk:
        # - num_vehicles reduced because vehicles are already captured
        #   by segmentation
        # - people and bikes remain important because they are vulnerable
        # - largest vehicle area captures proximity/blocking visibility
        yolo_risk = (
            0.06 * num_people
            + 0.035 * num_vehicles
            + 0.07 * num_bikes
            + 0.8 * largest_vehicle_area_ratio
        )

    visual_risk_raw = segmentation_risk + yolo_risk

    # Soft normalization avoids excessive saturation at 1.0.
    visual_risk_score = soft_normalize(visual_risk_raw)

    return {
        "visual_risk_raw": float(visual_risk_raw),
        "visual_risk_score": float(visual_risk_score),
    }


# ============================================================
# 3. Weather risk
# ============================================================

def compute_weather_risk(predicted_weather=None, confidence=0.0):
    """
    Converts predicted weather into a risk contribution.

    Weather is treated as a contextual modifier:
    - clear sunshine does not increase risk
    - sunrise slightly increases risk because of lighting/glare
    - cloudy weather moderately affects visibility
    - rain has the strongest weather penalty
    """

    if predicted_weather is None:
        return {
            "weather_score": 0.0,
            "weather_used": False,
        }

    weather = str(predicted_weather).lower()

    weather_weights = {
        "shine": 0.00,
        "sunrise": 0.10,
        "cloudy": 0.25,
        "rain": 0.65,
    }

    base_score = weather_weights.get(weather, 0.15)

    confidence = clamp(confidence)
    weather_score = base_score * confidence

    return {
        "weather_score": float(clamp(weather_score)),
        "weather_used": True,
    }


# ============================================================
# 4. Audio risk
# ============================================================

def compute_audio_risk(audio_result=None):
    """
    Audio is optional.

    If audio is available, it can add information about traffic sound,
    horns, sirens, or general traffic intensity.

    If no audio is available, it does not affect the score.
    """

    if audio_result is None:
        return {
            "audio_score": 0.0,
            "audio_used": False,
        }

    traffic_sound_score = audio_result.get("traffic_sound_score", 0.0)

    return {
        "audio_score": float(clamp(traffic_sound_score)),
        "audio_used": True,
    }


# ============================================================
# 5. Final danger score
# ============================================================

def compute_final_danger_score(
    visual_risk_score,
    weather_score=0.0,
    audio_score=0.0,
    weather_used=False,
    audio_used=False,
):
    """
    Combines available risk signals into a final danger score.

    Visual risk has the highest weight because it captures the immediate
    road scene: vehicles, pedestrians, cyclists, road layout, and proximity.

    Weather and audio are modifiers. They matter, but they should not
    dominate over what is directly visible in the scene.
    """

    visual_risk_score = clamp(visual_risk_score)
    weather_score = clamp(weather_score)
    audio_score = clamp(audio_score)

    # 🔥 UPDATED WEIGHTS (audio increased)

    if weather_used and audio_used:
        final_score = (
            0.60 * visual_risk_score
            + 0.15 * weather_score
            + 0.25 * audio_score
        )

    elif weather_used and not audio_used:
        final_score = (
            0.78 * visual_risk_score
            + 0.22 * weather_score
        )

    elif audio_used and not weather_used:
        final_score = (
            0.70 * visual_risk_score
            + 0.30 * audio_score
        )

    else:
        final_score = visual_risk_score

    final_score = clamp(final_score)

    if final_score < 0.33:
        danger_level = "Low"
    elif final_score < 0.66:
        danger_level = "Medium"
    else:
        danger_level = "High"

    return {
        "final_danger_score": float(final_score),
        "danger_level": danger_level,
    }