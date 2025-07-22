def chunk_text(text, max_chunk_words):
    """Split text into chunks at sentence boundaries, respecting sentence integrity.

    Args:
        text: The text to chunk
        max_chunk_words: Maximum number of words for each chunk

    Returns:
        List of text chunks, where each chunk respects sentence boundaries
    """
    # Extract sentences from the text
    sentences = extract_sentences(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If this sentence alone exceeds the limit, put it in its own chunk
        # (Don't split sentences - keep them whole)
        if sentence_word_count > max_chunk_words:
            # If we have a current chunk, finish it first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Add the long sentence as its own chunk
            chunks.append(sentence)
            continue

        # If adding this sentence would exceed the limit and we have content
        if current_size + sentence_word_count > max_chunk_words and current_chunk:
            # Finish current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_word_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_word_count

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_sentences(text):
    """Extract individual sentences from text."""
    # Split by periods, but be careful about abbreviations
    sentences = []

    # Simple sentence splitting - can be improved with more sophisticated logic
    parts = text.replace("\n", " ").split(".")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Add the period back
        sentence = part + "."

        # Skip very short fragments (likely abbreviations)
        if len(sentence.split()) >= 3:
            sentences.append(sentence)

    return sentences
