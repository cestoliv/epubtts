import numpy as np
import tempfile
import os
import soundfile as sf
from src.audio_utils import normalize_audio_level, append_audio_to_wav
from src.constants import NORMALIZE_AUDIO_LEVEL


def test_normalize_audio_to_target_level():
    """Test that audio is normalized to the correct target level."""
    test_audio = np.random.random(1000) * 0.05  # Very quiet audio
    normalized = normalize_audio_level(test_audio)
    
    # Calculate actual RMS level
    normalized_rms = np.sqrt(np.mean(normalized**2))
    target_rms = 10**(NORMALIZE_AUDIO_LEVEL / 20)  # -20dB = 0.1 linear
    
    # Should be close to target level (within 1% tolerance)
    assert abs(normalized_rms - target_rms) < 0.01


def test_append_audio_creates_file():
    """Test that append_audio_to_wav creates a new file correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.wav")
        audio_data = np.random.random(1000) * 0.1
        sample_rate = 22050
        
        append_audio_to_wav(test_file, audio_data, sample_rate, is_first_chunk=True)
        
        # File should exist
        assert os.path.exists(test_file)
        
        # File should have correct content
        loaded_audio, loaded_sr = sf.read(test_file)
        assert loaded_sr == sample_rate
        assert len(loaded_audio) == len(audio_data)


def test_append_audio_concatenates_correctly():
    """Test that audio chunks are properly concatenated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.wav")
        
        # Create and save first chunk
        audio1 = np.random.random(500) * 0.1
        append_audio_to_wav(test_file, audio1, 22050, is_first_chunk=True)
        
        # Append second chunk
        audio2 = np.random.random(300) * 0.1
        append_audio_to_wav(test_file, audio2, 22050, is_first_chunk=False)
        
        # Verify total length is sum of both chunks
        loaded_audio, _ = sf.read(test_file)
        expected_length = len(audio1) + len(audio2)
        assert len(loaded_audio) == expected_length


def test_audio_normalization_applied_during_append():
    """Test that audio is normalized when appending to file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.wav")
        
        # Create very quiet audio
        quiet_audio = np.random.random(1000) * 0.01
        append_audio_to_wav(test_file, quiet_audio, 22050, is_first_chunk=True)
        
        # Check that saved audio is normalized (louder than input)
        loaded_audio, _ = sf.read(test_file)
        loaded_rms = np.sqrt(np.mean(loaded_audio**2))
        original_rms = np.sqrt(np.mean(quiet_audio**2))
        
        # Normalized audio should be significantly louder
        assert loaded_rms > original_rms * 2