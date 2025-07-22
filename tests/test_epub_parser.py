import pytest
from unittest.mock import Mock, patch
from ebooklib import epub
from src.epub_parser import extract_chapters_from_epub


def test_file_not_found():
    """Test that missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_chapters_from_epub("nonexistent.epub")


@patch("src.epub_parser.epub.read_epub")
@patch("os.path.exists")
def test_extract_chapters_from_toc(mock_exists, mock_read_epub):
    """Test extracting chapters from table of contents."""
    mock_exists.return_value = True

    # Mock book with TOC
    mock_book = Mock()
    mock_link = Mock(spec=epub.Link)
    mock_link.title = "Chapter 1"
    mock_link.href = "chapter1.html"
    mock_book.toc = [mock_link]

    # Mock document
    mock_doc = Mock()
    mock_doc.file_name = "chapter1.html"
    mock_doc.get_content.return_value = (
        b"<html><body><p>Chapter content</p></body></html>"
    )
    mock_book.get_items_of_type.return_value = [mock_doc]
    mock_read_epub.return_value = mock_book

    with patch("builtins.print"):
        chapters = extract_chapters_from_epub("test.epub")

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter 1"
    assert "Chapter content" in chapters[0]["content"]


@patch("src.epub_parser.epub.read_epub")
@patch("os.path.exists")
def test_extract_chapters_fallback(mock_exists, mock_read_epub):
    """Test fallback when no TOC is available."""
    mock_exists.return_value = True

    # Mock book with empty TOC
    mock_book = Mock()
    mock_book.toc = []

    # Mock document with chapter heading
    mock_doc = Mock()
    mock_doc.file_name = "content.html"
    mock_doc.get_content.return_value = b"""
    <html><body>
    <h1>Chapter Title</h1>
    <p>Chapter content here.</p>
    </body></html>
    """
    mock_book.get_items_of_type.return_value = [mock_doc]
    mock_read_epub.return_value = mock_book

    with patch("builtins.print"):
        chapters = extract_chapters_from_epub("test.epub")

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter Title"
