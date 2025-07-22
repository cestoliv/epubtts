import os
import numpy as np
import soundfile as sf
from .constants import NORMALIZE_AUDIO_LEVEL


def normalize_audio_level(audio_data, target_db=NORMALIZE_AUDIO_LEVEL):
    """Normalize audio to target RMS level in dB."""
    if len(audio_data) == 0:
        return audio_data

    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio_data**2))
    if rms == 0:
        return audio_data

    # Convert to dB
    current_db = 20 * np.log10(rms)

    # Calculate gain needed to reach target level
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain and prevent clipping
    normalized_audio = audio_data * gain_linear
    max_val = np.max(np.abs(normalized_audio))
    if max_val > 0.95:  # Prevent clipping with small headroom
        normalized_audio = normalized_audio * (0.95 / max_val)

    return normalized_audio


def append_audio_to_wav(output_file, audio_data, sample_rate, is_first_chunk=False):
    """Append audio data to a WAV file. If first chunk, create new file."""
    # Normalize audio level before processing
    normalized_audio = normalize_audio_level(audio_data)

    if is_first_chunk:
        sf.write(output_file, normalized_audio, sample_rate)
        return

    if not os.path.exists(output_file):
        sf.write(output_file, normalized_audio, sample_rate)
        return

    # Read existing audio and append new data
    existing_audio, existing_sr = sf.read(output_file)
    if existing_sr != sample_rate:
        print(
            f"Warning: Sample rate mismatch - existing: {existing_sr}, new: {sample_rate}"
        )

    # Ensure both audio arrays are 1D
    if len(existing_audio.shape) > 1:
        existing_audio = existing_audio.flatten()
    if len(normalized_audio.shape) > 1:
        normalized_audio = normalized_audio.flatten()

    # Concatenate and write
    combined_audio = np.concatenate([existing_audio, normalized_audio])
    sf.write(output_file, combined_audio, sample_rate)
