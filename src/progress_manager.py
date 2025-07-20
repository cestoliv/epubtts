import os
import time
import pickle

def save_progress(progress_file, chapter_index, chunk_index, total_chunks, selected_chapters=None):
    """Save progress to a file for resume functionality."""
    progress_data = {
        'chapter_index': chapter_index,
        'chunk_index': chunk_index,
        'total_chunks': total_chunks,
        'timestamp': time.time(),
        'selected_chapters': selected_chapters
    }
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)

def load_progress(progress_file):
    """Load progress from a file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None
