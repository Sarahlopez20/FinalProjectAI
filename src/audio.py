# ============================================================
# AUDIO MODULE
# Optional audio-risk component.
# ============================================================

from pathlib import Path
import librosa
import numpy as np


# ============================================================
# 1. Analyze traffic audio
# ============================================================

def analyze_audio_file(audio_path):
    """
    Returns a simple audio risk score from an audio file.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    if len(y) == 0:
        return {
            "traffic_sound_score": 0.0,
            "audio_rms": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_used": False,
            "audio_file": str(audio_path.name),
        }

    # Root mean square energy: loudness proxy
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    # Spectral centroid: brightness proxy
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(spectral_centroid))

    # Normalize
    rms_score = min(mean_rms / 0.10, 1.0)
    centroid_score = min(mean_centroid / 4000.0, 1.0)

    traffic_sound_score = 0.65 * rms_score + 0.35 * centroid_score
    traffic_sound_score = float(max(0.0, min(1.0, traffic_sound_score)))

    return {
        "traffic_sound_score": traffic_sound_score,
        "audio_rms": mean_rms,
        "audio_spectral_centroid": mean_centroid,
        "audio_used": True,
        "audio_file": str(audio_path.name),
    }


# ============================================================
# 2. Safe wrapper
# ============================================================

def analyze_audio_optional(audio_path=None):
    """
    If audio exists, analyze it.
    If not, return None so the final pipeline skips audio.
    """

    if audio_path is None:
        return {
            "traffic_sound_score": 0.0,
            "audio_rms": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_used": False,
            "audio_file": None,
        }

    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "traffic_sound_score": 0.0,
            "audio_rms": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_used": False,
            "audio_file": None,
        }

    try:
        return analyze_audio_file(audio_path)
    except Exception as e:
        print(f"Audio error for {audio_path}: {e}")
        return {
            "traffic_sound_score": 0.0,
            "audio_rms": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_used": False,
            "audio_file": str(audio_path.name),
        }