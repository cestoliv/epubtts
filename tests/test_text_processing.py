from src.text_processing import chunk_text, extract_sentences


def test_extract_sentences_basic():
    """Test basic sentence extraction with proper length sentences."""
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    sentences = extract_sentences(text)
    
    assert len(sentences) == 3
    assert "This is the first sentence." in sentences
    assert "This is the second sentence." in sentences
    assert "This is the third sentence." in sentences


def test_extract_sentences_filters_short():
    """Test that short sentences (< 3 words) are filtered out."""
    text = "Short. This is a longer sentence. Ok."
    sentences = extract_sentences(text)
    
    # Only the longer sentence should remain
    assert len(sentences) == 1
    assert "This is a longer sentence." in sentences


def test_chunk_text_within_limit():
    """Test chunking when text fits in one chunk."""
    text = "This is a short sentence. Another short one."
    chunks = chunk_text(text, max_chunk_words=20)
    
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_exceeds_limit():
    """Test chunking when text needs multiple chunks."""
    text = "This is first sentence. This is second sentence with many more words than allowed."
    chunks = chunk_text(text, max_chunk_words=5)
    
    # Should create multiple chunks due to word limit
    assert len(chunks) >= 2
    # Verify content is preserved
    combined = ' '.join(chunks)
    assert "first sentence" in combined
    assert "second sentence" in combined


def test_chunk_text_preserves_sentences():
    """Test that sentences are never split across chunks."""
    text = "First sentence here. Second sentence with more words. Third sentence also here."
    chunks = chunk_text(text, max_chunk_words=6)
    
    # Each chunk should end with a period (complete sentences)
    for chunk in chunks:
        assert chunk.strip().endswith('.')
        # Should not have partial sentences
        assert '. ' not in chunk[:-1]  # No sentence breaks except at the end