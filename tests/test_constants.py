from src.constants import (
    DEFAULT_VOICE,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_WPM,
    NORMALIZE_AUDIO_LEVEL
)


def test_constants_basic_validation():
    """Test that all constants have expected types and reasonable values."""
    assert isinstance(DEFAULT_VOICE, str) and len(DEFAULT_VOICE) > 0
    assert isinstance(DEFAULT_CHUNK_WORDS, int) and DEFAULT_CHUNK_WORDS > 0
    assert isinstance(DEFAULT_WPM, int) and DEFAULT_WPM > 0
    assert isinstance(NORMALIZE_AUDIO_LEVEL, float) and NORMALIZE_AUDIO_LEVEL < 0


def test_default_values():
    """Test specific default values."""
    assert DEFAULT_CHUNK_WORDS == 500
    assert DEFAULT_WPM == 150
    assert NORMALIZE_AUDIO_LEVEL == -20.0
    assert DEFAULT_VOICE.endswith('.wav')