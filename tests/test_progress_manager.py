import tempfile
import os
from src.progress_manager import save_progress, load_progress


def test_save_and_load_progress():
    """Test basic save and load functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_file = os.path.join(temp_dir, "test_progress.pkl")
        
        # Save progress
        save_progress(progress_file, 2, 5, 10, [1, 2, 3])
        
        # Load progress
        data = load_progress(progress_file)
        
        assert data is not None
        assert data['chapter_index'] == 2
        assert data['chunk_index'] == 5
        assert data['total_chunks'] == 10
        assert data['selected_chapters'] == [1, 2, 3]
        assert 'timestamp' in data


def test_load_nonexistent_progress():
    """Test loading progress from non-existent file."""
    data = load_progress("nonexistent_file.pkl")
    assert data is None


def test_progress_overwrites():
    """Test that saving overwrites existing progress."""
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_file = os.path.join(temp_dir, "test_progress.pkl")
        
        # Save first progress
        save_progress(progress_file, 1, 2, 5, [1])
        
        # Save updated progress
        save_progress(progress_file, 3, 4, 8, [1, 2, 3])
        
        # Should have the updated values
        data = load_progress(progress_file)
        assert data['chapter_index'] == 3
        assert data['chunk_index'] == 4